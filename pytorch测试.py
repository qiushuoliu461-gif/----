import torch

print("PyTorch版本:", torch.__version__)
print("显卡是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("你的显卡型号:", torch.cuda.get_device_name(0))
    # 在显卡上创建一个矩阵测试一下
    x = torch.rand(3, 3).cuda()
    print("测试生成的显卡矩阵:\n", x)