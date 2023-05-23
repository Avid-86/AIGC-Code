# 开发时间: 2023/5/17 22:14
import torchvision
LOAD_CIFAR = True
DOWNLOAD_CIFAR = True
train_data = torchvision.datasets.CIFAR10(
    root='data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_CIFAR,
)
