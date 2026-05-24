import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 【配置区】：请根据您在 train_resnet18.py 中最终保留的网络结构，保留其中一种导入方式
# ---------------------------------------------------------
# 方式 A（如果您目前仍在使用本地编写的 ResNet18）:
from model.model_resnet34 import ResNet34

# 方式 B（如果您使用了 torchvision 官方的模型库）:
# from torchvision.models import resnet18
# import torch.nn as nn
# ---------------------------------------------------------

def main():
    # 自动探测并挂载您的 RTX 4090
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 物理路径映射（确保与训练输出的目录严格一致）
    json_path = './ResNet34/weights_flower/class_indices.json'
    weights_path = './ResNet34/weights_flower/best_model.pth'
    
    # 您需要准备一张测试图片，并放在该路径下
    img_path = './test_image.jpg'

    # 2. 读取类别索引字典
    assert os.path.exists(json_path), f"字典文件不存在，请检查路径: {json_path}"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    num_classes = len(class_indict)

    # 3. 实例化模型底座（代码逻辑必须与训练时结构完全镜像）
    # 如果您使用方式 A:
    model = ResNet34(num_classes=num_classes).to(device)
    
    # 如果您使用方式 B（请取消下方三行注释，并注释掉方式 A）:
    # model = resnet18()
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = model.to(device)

    # 4. 加载训练好的最佳权重
    assert os.path.exists(weights_path), f"权重文件不存在: {weights_path}"
    # map_location 参数确保无论当初模型在哪个显卡上训练，都能正确加载到当前 device
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 核心步骤：必须切换到评估模式，锁定 Batch Normalization 和 Dropout
    model.eval()

    # 5. 图像预处理流水线（工程铁律：推理时禁止使用任何随机数据增强算子）
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 6. 读取并预处理输入张量
    assert os.path.exists(img_path), f"找不到测试图片: {img_path}"
    img = Image.open(img_path).convert('RGB')
    
    # 拷贝一份原图，用于最后的绘图展示
    plt_img = img.copy()

    # 扩充 Batch 维度：[Channel, Height, Width] -> [1, Channel, Height, Width]
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 7. 正向传播计算
    with torch.no_grad():  # 显式关闭计算图追踪，大幅节省显存并加速推理
        # 输入模型并压缩掉多余的 Batch 维度
        output = torch.squeeze(model(img_tensor.to(device)))
        # 使用 Softmax 将输出的 Logits 转化为 [0, 1] 之间的概率分布
        predict = torch.softmax(output, dim=0)
        # 获取最大概率所在的张量索引
        predict_cla = torch.argmax(predict).item()

    predict_label = class_indict[str(predict_cla)]
    predict_prob = predict[predict_cla].item()

    # 8. 终端定量结果输出
    print("-" * 50)
    print(f"输入测试图像: {img_path}")
    print(f"网络预测类别: {predict_label}")
    print(f"预测置信概率: {predict_prob * 100:.2f}%")
    print("-" * 50)

    # 9. 可视化呈现（弹出包含预测结果的图片窗口）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(6, 6))
    plt.imshow(plt_img)
    plt.title(f"ResNet34_Predict: {predict_label}\nProb: {predict_prob:.4f}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    save_path = "ResNet34_predict_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__':
    main()
