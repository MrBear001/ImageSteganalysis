from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from LWENet import lwenet
from torch.backends import cudnn
import numpy as np
import random
from time import *
import utils
import os
from torchsummary import summary

# ==============================================================================
# 新增: 用于确保实验可复现性的函数
# ==============================================================================
def set_seed(seed):
    """
    Sets the random seeds for Python, Numpy and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # 关键步骤: 禁用 cudnn 的自动调优功能，强制使用确定性算法
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"Random seed set to {seed}, and cuDNN is configured for deterministic behavior.")

# ==============================================================================
#                               主代码开始
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch LWENet Training')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='wd',
                    help='weight_decay (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# 修改: 将 seed 的默认值改为 42，并更新帮助信息
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default='checkpoints',
                    help='directory to save model checkpoints (default: checkpoints)')

# 解析参数
args = parser.parse_args()

# ==============================================================================
# 修改: 调用新的 set_seed 函数，并移除旧的种子设置代码
# ==============================================================================
set_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
# 旧的种子设置代码已被移除
# ==============================================================================

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# 训练集路径
train_cover_path = "/root/autodl-tmp/new-dataset-huafen/HILL/0.4bpp/train/cover"
train_stego_path = "/root/autodl-tmp/new-dataset-huafen/HILL/0.4bpp/train/stego"

# 验证集路径
valid_cover_path = "/root/autodl-tmp/new-dataset-huafen/HILL/0.4bpp/valid/cover"
valid_stego_path = "/root/autodl-tmp/new-dataset-huafen/HILL/0.4bpp/valid/stego"

# 测试集路径
test_cover_path = "/root/autodl-tmp/new-dataset-huafen/HILL/0.4bpp/test/cover"
test_stego_path = "/root/autodl-tmp/new-dataset-huafen/HILL/0.4bpp/test/stego"

print('torch version: ', torch.__version__)
print('train_cover_path = ', train_cover_path)
print('valid_cover_path = ', valid_cover_path)
print('test_cover_path = ', test_cover_path)
print('train_batch_size = ', args.batch_size)
print('test_batch_size = ', args.test_batch_size)

# 训练集的数据增强和转换
train_transform = transforms.Compose([utils.AugData(), utils.ToTensor()])

# 验证集和测试集不需要数据增强
valid_test_transform = utils.ToTensor()

# 使用 utils.DatasetPair 来加载所有数据集
train_data = utils.DatasetPair(train_cover_path, train_stego_path, train_transform)
valid_data = utils.DatasetPair(valid_cover_path, valid_stego_path, valid_test_transform)
test_data = utils.DatasetPair(test_cover_path, test_stego_path, valid_test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

model = lwenet()
print(model)
if args.cuda:
    model.cuda()
    print(summary(model, (1, 256, 256)))


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
model.apply(initWeights)

params = model.parameters()
params_wd, params_rest = [], []
for param_item in params:
    if param_item.requires_grad:
        (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

param_groups = [{'params': params_wd, 'weight_decay': args.weight_decay},
                {'params': params_rest}]

optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

DECAY_EPOCH = [80, 140, 180]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)

def train(epoch):
    total_loss = 0
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    print(f"Current learning rate: {lr_train}")

    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        if args.cuda:
            data, label = batch_data['images'].cuda(), batch_data['labels'].cuda()
        else:
            data, label = batch_data['images'], batch_data['labels']

        data, label = Variable(data), Variable(label)

        batch_size = data.shape[0]
        datas = data.view(batch_size * 2, 1, 256, 256)
        labels = label.view(batch_size * 2)

        optimizer.zero_grad()
        output = model(datas)
        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % args.log_interval == 0:
            b_pred = output.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()
            b_accu = b_correct / labels.size(0)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * batch_size, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), b_accu, loss.item()))

    print('train Epoch: {}\tavgLoss: {:.6f}'.format(epoch, total_loss / len(train_loader)))
    scheduler.step()

def test():
    model.eval()
    test_loss = 0
    correct = 0.
    total_samples = 0
    with torch.no_grad():
        for batch_data in test_loader:
            if args.cuda:
                data, target = batch_data['images'].cuda(), batch_data['labels'].cuda()
            else:
                data, target = batch_data['images'], batch_data['labels']

            batch_size = data.shape[0]
            data = data.view(batch_size * 2, 1, 256, 256)
            target = target.view(batch_size * 2)
            total_samples += target.size(0)

            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_samples
    accuracy = 100. * correct / total_samples
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, total_samples, accuracy))
    return accuracy, test_loss

def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
    total_samples = 0
    with torch.no_grad():
        for batch_data in valid_loader:
            if args.cuda:
                data, target = batch_data['images'].cuda(), batch_data['labels'].cuda()
            else:
                data, target = batch_data['images'], batch_data['labels']

            batch_size = data.shape[0]
            data = data.view(batch_size * 2, 1, 256, 256)
            target = target.view(batch_size * 2)
            total_samples += target.size(0)

            data, target = Variable(data), Variable(target)
            output = model(data)
            valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= total_samples
    accuracy = 100. * correct / total_samples
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        valid_loss, correct, total_samples, accuracy))
    return accuracy, valid_loss

def sum(pred, target):
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i] + target[i])
    return l1.count(0), l1.count(2), l1.count(0) + l1.count(2)


t1 = time()

# 确保保存目录存在
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print(f"Created directory for saving checkpoints: {args.save_dir}")

for epoch in range(1, args.epochs + 1):
    train(epoch)
    valid()
    test()

    # 保存模型权重
    save_filename = f'lwenet_epoch_{epoch}.pkl'
    save_path = os.path.join(args.save_dir, save_filename)
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
    print(f'Epoch {epoch}: Model weights saved to {save_path}')

t2 = time()
print(f"Total training time: {t2 - t1:.2f} seconds")