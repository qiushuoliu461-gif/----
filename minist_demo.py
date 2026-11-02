import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 2. 检查GPU是否可用 (再次确认5060在干活)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# 3. 数据预处理与加载
# 训练集：加上随机旋转和平移（数据增强） 不断的给他图片和答案，调参数
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),# 转成 PyTorch 张量
    transforms.Normalize((0.1307,), (0.3081,))
])  # 新值 = （原值-0.1307）/0.3081

# 测试集：绝对不能加数据增强！只能做基础转换 只检查结果，没见过数据
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载训练集（用 train_transform）
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)

# 下载测试集（用 test_transform）
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 每轮把图片顺序打乱
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False) # 每轮不打乱


# 4. 搭建一个极简的卷积神经网络 (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 特征提取层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # 分类层 (28x28 经过池化变成 14x14)
        self.fc = nn.Linear(16 * 14 * 14, 10)  # 输出10个数字(0-9)的类别的得分

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)  # 展平
        x = self.fc(x)
        return x # 返回的是十个得分，不是数字，也不是概率


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，统计学里常用于衡量两个概率分布的差异
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 开始训练（只需 20 轮， 5060 会飞快跑完）
epochs = 20
loss_history = []

print("开始训练...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 300 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

# 6. 模型评估与可视化
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"\n训练结束！在 10000 张测试集图片上的准确率: {accuracy:.2f}%")

# --- 画图环节 ---
plt.figure(figsize=(10, 4))

# 左图：Loss下降曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='r')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')

# 右图：随机抽 5 张图片看预测结果
plt.subplot(1, 2, 2)
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

for i in range(5):
    plt.subplot(1, 5, i + 1)  # 位置调整
    img = images[i].cpu().numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}")
    plt.axis('off')

plt.tight_layout()
plt.show()
# 把训练好的模型存到硬盘里
torch.save(model.state_dict(), "mnist_model.pth")