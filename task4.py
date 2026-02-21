import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from thop import profile

# ---------------------- 1. 数据增强与加载 ----------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ---------------------- 2. 结构优化：带残差连接的改进CNN ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=20):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * np.pi))
    return LambdaLR(optimizer, lr_lambda)

# ---------------------- 4. 训练与评估 ----------------------
def train(net, trainloader, criterion, optimizer, scheduler, use_mixup=True, alpha=0.2):
    net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    
    return running_loss / len(trainloader)

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ---------------------- 5. 主函数：对比实验 ----------------------
if __name__ == '__main__':
    # 计算参数量与FLOPs
    net = ImprovedCNN().cuda()
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(net, inputs=(input, ))
    print(f"参数量: {params/1e6:.2f} M, FLOPs: {flops/1e6:.2f} M")

    # 对比实验：有无Warmup+Mixup
    for use_trick in [False, True]:
        net = ImprovedCNN().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        scheduler = get_warmup_scheduler(optimizer) if use_trick else None
        
        print(f"\n=== 使用Warmup+Mixup: {use_trick} ===")
        for epoch in range(20):
            loss = train(net, trainloader, criterion, optimizer, scheduler, use_mixup=use_trick)
            acc = test(net, testloader)
            print(f'Epoch {epoch+1}, Loss: {loss:.3f}, Test Acc: {acc:.2f}%')
            
def train(net, trainloader, criterion, optimizer, scheduler, use_mixup=True, alpha=0.2):
    net.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return running_loss / len(trainloader)

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ---------------------- 5. 主函数：对比实验 ----------------------
if __name__ == '__main__':
    # 计算参数量与FLOPs
    net = ImprovedCNN().cuda()
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(net, inputs=(input, ))
    print(f"参数量: {params/1e6:.2f} M, FLOPs: {flops/1e6:.2f} M")

    # 对比实验：有无Warmup+Mixup
    for use_trick in [False, True]:
        net = ImprovedCNN().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        scheduler = get_warmup_scheduler(optimizer) if use_trick else None
        
        print(f"\n=== 使用Warmup+Mixup: {use_trick} ===")
        for epoch in range(20):
            loss = train(net, trainloader, criterion, optimizer, scheduler, use_mixup=use_trick)
            acc = test(net, testloader)
            print(f'Epoch {epoch+1}, Loss: {loss:.3f}, Test Acc: {acc:.2f}%')

    # 可视化预测结果
    def imshow(img):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)
        img = img * std + mean
        npimg = img.numpy()
        npimg = np.clip(npimg, 0.0, 1.0)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = net(images.cuda())
    _, predicted = torch.max(outputs, 1)

    imshow(torchvision.utils.make_grid(images[:8]))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(8)))