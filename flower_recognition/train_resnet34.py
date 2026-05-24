import os
import argparse
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import torch.nn as nn

from dataset import MyDataSet
from model.model_resnet34 import ResNet34  # 导入 ResNet34 模型
from utils import train_one_epoch, evaluate, read_data


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_config = {
        'flower': {
            'Dataset': MyDataSet,
            'train_data_path': "./dataset/flower/train",
            'test_data_path': "./dataset/flower/val",
            'in_c': 3,
            'num_classes': 14
        },
    }

    test_data_path = dataset_config[args.dataset]['test_data_path']
    train_data_path = dataset_config[args.dataset]['train_data_path']

    output_dir = f"./ResNet34/weights_{args.dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_images_path, train_images_label, val_images_path, val_images_label = read_data(test_data_path,train_data_path, output_dir, args.rate)

    # 图像预处理：适配 256x256
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandAugment(), # 核心杀手锏：激活 14 种随机物理扰动
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整到固定尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 实例化训练数据集和验证数据集
    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=transform_train)
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=transform_test)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers per process')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    # 创建 ResNet34 模型实例
    model = ResNet34(num_classes=args.num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"---------开始第{epoch + 1}/{args.epochs}轮训练，本轮学习率为：{optimizer.param_groups[0]['lr']}---------")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader,
            device=device, epoch=epoch)

        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
        print(f"验证损失: {val_loss:.3f}, 验证精确度: {val_acc:.3f}")

        elapsed = (time.time() - start_time) / 60
        print(f'本轮训练累计用时: {elapsed:.2f} min')

        # 如果当前验证精度比最佳精度高，则保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最佳模型权重
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pth"))
            print(f"保存了新的最佳模型，精确度: {best_val_acc:.3f}")

    print(f"训练完成，最佳验证精度为: {best_val_acc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=30)  # 增加训练轮次
    parser.add_argument('--batch-size', type=int, default=32)  # 调整 batch size 以适应更大图像
    parser.add_argument('--lr', type=float, default=0.0001)  # 降低学习率以提高稳定性
    parser.add_argument('--dataset', type=str, default='flower')
    parser.add_argument('--rate', type=float, default=0.8, help='dataset size')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
