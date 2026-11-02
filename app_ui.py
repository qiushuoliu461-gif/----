import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np


# 1. 定义模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x


# 2. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()


# 3. 定义预测函数
def predict_digit(image_data):
    if image_data is None:
        return "请先在画板上写一个数字哦！"

    # 无论传进来是什么字典，我们直接强行把它转成 PIL 图像
    # 如果是 ImageEditor 传进来的字典，拿里面的 'composite'
    if isinstance(image_data, dict):
        if 'composite' in image_data and image_data['composite'] is not None:
            image = image_data['composite']
        else:
            return "请先画图！"
    else:
        # 如果是直接传的 numpy 数组或 PIL 图像
        image = image_data

    # 确保它是 PIL Image 对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 压缩成灰度图（自动忽略你选的红色）
    image = image.convert('L')

    # 反转颜色（白底黑字 -> 黑底白字）
    image = ImageOps.invert(image)

    # 缩放到 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return f"识别结果：{predicted.item()}"


# 4. 搭建 UI 界面 (使用 Blocks 布局，更稳定)
with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("## 🔥 PyTorch 手写数字识别器")
    gr.Markdown("这是基于 CNN 训练的交互界面。用鼠标在画板上写一个数字（颜色随意），然后点击下方的【提交】。")

    with gr.Row():
        with gr.Column():
            # 换成了 gr.Paint，这玩意儿最稳定，不会报错
            canvas = gr.Paint(label="用鼠标在这里手写数字", type="pil",
                              brush=gr.Brush(colors=["#000000"], default_color="#000000", default_size=20))
            submit_btn = gr.Button("提交识别", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="模型预测结果", interactive=False)

    # 绑定按钮点击事件
    submit_btn.click(fn=predict_digit, inputs=canvas, outputs=output_text)

# 5. 启动
if __name__ == "__main__":
    demo.launch()