import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 导入您自己编写的 ResNet34 封装类
from model.model_resnet34 import ResNet34

# 导入 CAM 核心模块
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 物理路径映射（确保与训练时保存的路径一致）
    json_path = './ResNet34/weights_flower/class_indices.json'
    weights_path = './ResNet34/weights_flower/best_model.pth'
    
    img_path = './test_image.jpg'  # 请将此处替换为您想测试的图片路径

    # 2. 读取类别索引字典
    assert os.path.exists(json_path), f"找不到字典文件: {json_path}"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    num_classes = len(class_indict)

    # 3. 实例化模型并加载最佳权重
    #model = ResNet34(num_classes=num_classes).to(device)
    model = ResNet34(num_classes=num_classes).to(device)
    assert os.path.exists(weights_path), f"找不到权重文件: {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 4. 指定目标特征层（核心物理机制）
    # ResNet 提取的高级语义特征聚集在最后一个卷积 block 中
    # 因为在您的类中，底座被命名为 self.resnet34，因此路径为 model.resnet34.layer4[-1]
    target_layers = [model.resnet34.layer4[-1]]

    # 5. 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 读取图片
    assert os.path.exists(img_path), f"找不到测试图片: {img_path}"
    img = Image.open(img_path).convert('RGB')
    input_tensor = data_transform(img).unsqueeze(0).to(device)

    # 为了将热力图叠加在原图上，需要一张归一化到 [0,1] 的 numpy 格式原图
    img_resize = img.resize((256, 256))
    rgb_img = np.float32(img_resize) / 255.0

    # 6. 初始化 Grad-CAM 引擎
    cam = GradCAM(model=model, target_layers=target_layers)

    # 7. 生成注意力掩码 (targets=None 意味着默认对模型预测概率最高的类别求导)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    # 提取 batch 中第一张图片的掩码
    grayscale_cam = grayscale_cam[0, :] 

    # 8. 将掩码颜色映射并叠加到原图
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 9. 绘制学术对比图
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_resize)
    plt.title("Original Image", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title("Grad-CAM Attention Map", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("R34_grad_cam_result.png", dpi=300, bbox_inches='tight')
    print(">>> 成功！高分辨率热力图已保存为 grad_cam_result.png")
    plt.show()

if __name__ == '__main__':
    main()
