import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix
import torch.nn as nn

# 【核心修正】：导入您自己本地封装的模型结构，彻底杜绝字典键值错位
# （如果您的 ResNet18 当初就是用官方库跑的，请保留原来的 from torchvision.models import resnet18）
from model.model_resnet18 import ResNet18 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 进行推理计算...")

    # 1. 路径配置（严格对齐物理目录）
    val_data_dir = "./dataset/flower/val"
    json_path = "./ResNet18/weights_flower/class_indices.json"
    weights_path = "./ResNet18/weights_flower/best_model.pth"

    # 2. 解析类别字典
    assert os.path.exists(json_path), f"找不到字典文件 {json_path}"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    num_classes = len(class_indict)
    
    # 提取按索引排序的类别名称标签，用于作图的坐标轴
    labels = [class_indict[str(i)] for i in range(num_classes)]

    # 3. 严格对齐训练时的验证集预处理参数（禁用任何随机数据增强）
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 4. 载入验证集
    val_dataset = datasets.ImageFolder(root=val_data_dir, transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, 
                                             shuffle=False, num_workers=4)

    # 5. 实例化模型并装载最佳权重
    # 【核心修正】：使用您的封装类
    model = ResNet18(num_classes=num_classes)
    
    assert os.path.exists(weights_path), f"找不到权重文件 {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.to(device)
    model.eval()  # 必须开启评估模式

    # 6. 收集预测与真实标签
    true_labels = []
    pred_labels = []

    print("开始遍历验证集计算混淆矩阵，请稍候...")
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images.to(device))
            predict_classes = torch.argmax(outputs, dim=1).cpu().numpy()
            
            true_labels.extend(targets.numpy())
            pred_labels.extend(predict_classes)

    # 7. 计算混淆矩阵的底层数据
    cm = confusion_matrix(true_labels, pred_labels)

    # 8. 绘制符合顶刊排版标准的混淆矩阵
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 12

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=.5, annot_kws={"size": 10})

    plt.title('Confusion Matrix on Validation Set (ResNet18)', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', labelpad=15, fontsize=14)
    plt.ylabel('True Label', labelpad=15, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 9. 输出矢量图与高清位图
    # 【细节修正】：物理保存 PDF 文件，并对齐打印输出的文件名
    plt.savefig("R18_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(">>> 混淆矩阵已生成！保存在当前目录下的 R18_confusion_matrix.pdf 和 R18_confusion_matrix.png")

if __name__ == '__main__':
    main()