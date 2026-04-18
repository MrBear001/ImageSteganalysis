import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import utils  # 确保 utils.py 在同一目录
from LWENet import lwenet  # 确保 LWENet.py 在同一目录

def test_metrics(args):
    """
    主测试函数，用于计算准确率、AUC并绘制ROC曲线
    """
    # 检查 CUDA 是否可用
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using {device} for testing.")

    # 1. 准备数据集
    print(f"Loading test data from:")
    print(f"  Cover path: {args.cover_dir}")
    print(f"  Stego path: {args.stego_dir}")

    test_transform = utils.ToTensor()
    test_data = utils.DatasetPair(args.cover_dir, args.stego_dir, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.cuda else False
    )

    # 2. 加载模型架构和权重
    print("Initializing model architecture...")
    model = lwenet()

    if not os.path.exists(args.weights):
        print(f"Error: Weight file not found at '{args.weights}'")
        return

    print(f"Loading trained weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # 3. 开始评估
    model.eval()

    # 初始化用于存储所有结果的变量
    all_labels = []  # 用于AUC计算
    all_scores = []  # 用于AUC计算
    correct = 0  # 用于准确率计算
    total_samples = 0  # 用于准确率计算

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            data, target = batch_data['images'].to(device), batch_data['labels'].to(device)

            # 将成对的数据展平以适应模型输入
            batch_size = data.shape[0]
            data = data.view(batch_size * 2, 1, 256, 256)
            target = target.view(batch_size * 2)

            # 模型前向传播
            output = model(data)

            # --- AUC 计算部分 ---
            probabilities = F.softmax(output, dim=1)
            scores = probabilities[:, 1]
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

            # --- 准确率计算部分 ---
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)

            if (i + 1) % 50 == 0:
                print(f"Processing batch {i + 1}/{len(test_loader)}")

    # 4. 计算并报告最终指标
    print("\nCalculating final metrics...")

    # 计算准确率
    accuracy = 100. * correct / total_samples

    # 计算 ROC 和 AUC
    fpr, tpr, thresholds = roc_curve(np.array(all_labels), np.array(all_scores), pos_label=1)
    roc_auc = auc(fpr, tpr)

    print("\n" + "=" * 45)
    print("        Comprehensive Test Results         ")
    print("=" * 45)
    print(f'Model: {args.weights}')
    print(f'Test Dataset Size: {total_samples} images')
    print("-" * 45)
    print(f'Accuracy: {correct}/{total_samples} ({accuracy:.4f}%)')
    print(f'AUC Score: {roc_auc:.6f}')
    print("=" * 45)

    # 5. 绘制并保存 ROC 曲线图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(args.output_file)
    print(f"ROC curve plot saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LWENet Comprehensive Tester (Accuracy & AUC)')

    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the trained model weights (.pkl file)')
    parser.add_argument('--cover-dir', type=str, required=True,
                        help='Path to the directory of cover images for testing')
    parser.add_argument('--stego-dir', type=str, required=True,
                        help='Path to the directory of stego images for testing')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for testing (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA for testing')
    parser.add_argument('--output-file', type=str, default='roc_curve_metrics.png',
                        help='File name to save the ROC curve plot (default: roc_curve_metrics.png)')

    args = parser.parse_args()
    test_metrics(args)