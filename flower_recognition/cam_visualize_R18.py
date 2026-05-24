import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 【核心修改 1】：导入您本地编写的 ResNet18 封装类
from model.model_resnet18 import ResNet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 生成 ResNet18 的热力图...")

    # 【核心修改 2】：物理路径严格指向 ResNet18 的输出目录
    json_path = './ResNet18/weights_flower/class_indices.json'
    weights_path = './ResNet18/weights_flower/best_model.pth'
    
    img_path = './test_image.jpg'  # 您那张康乃馨测试图片的路径

    # 读取类别索引字典
    assert os.path.exists(json_path), f"找不到字典文件: {json_path}"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    num_classes = len(class_indict)

    # 【核心修改 3】：实例化 ResNet18 模型并加载对应的最佳权重
    model = ResNet18(num_classes=num_classes).to(device)
    assert os.path.exists(weights_path), f"找不到权重文件: {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 【核心修改 4】：指定目标特征层
    # 假设您的 model_resnet18.py 中底座被命名为 self.resnet18
    # 如果您的底座命名不同（例如 self.model），请在此处同步修改
    try:
        target_layers = [model.resnet18.layer4[-1]]
    except AttributeError:
        # 兼容性处理：如果是直接继承的 torchvision 原生结构
        target_layers = [model.layer4[-1]]

    # 数据预处理 (推理时禁止随机增强)
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 读取图片并预处理
    assert os.path.exists(img_path), f"找不到测试图片: {img_path}"
    img = Image.open(img_path).convert('RGB')
    input_tensor = data_transform(img).unsqueeze(0).to(device)

    # 归一化原图用于热力图叠加
    img_resize = img.resize((256, 256))
    rgb_img = np.float32(img_resize) / 255.0

    # 初始化 Grad-CAM 引擎
    cam = GradCAM(model=model, target_layers=target_layers)

    # 生成注意力掩码 (targets=None 默认对模型预测概率最高的类别求导)
    # 对于之前犯错的 ResNet18，这里求导的类别自动就是 "water_lily"
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    # 提取第一张图片的掩码
    grayscale_cam = grayscale_cam[0, :] 

    # 将掩码叠加到原图
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 绘制学术对比图
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_resize)
    plt.title("Original Image", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title("Grad-CAM (ResNet18)", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    # 【细节修改】：修改输出文件名，防止覆盖 ResNet34 的图
    plt.savefig("R18_grad_cam_result.png", dpi=300, bbox_inches='tight')
    print(">>> 成功！ResNet18 的高分辨率热力图已保存为 R18_grad_cam_result.png")
    plt.show()

if __name__ == '__main__':
    main()