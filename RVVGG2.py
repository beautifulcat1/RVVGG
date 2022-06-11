# 训练vgg，此处没有用vgg16，用的vgg8
import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(4096, 10))

if __name__ == '__main__' :
    conv_arch = ((1, 64), (1, 128), (1, 256), (1, 512), (1, 512))
    net = vgg(conv_arch)
    lr, num_epochs, batch_size, num_class = 0.05, 10, 16, 10
    root_dir = r"D:\test"
    train_iter, test_iter = d2l.load_my_dataset(root_dir, batch_size, num_class, shuffle = True, resize=(224,224))#将RVVGG0封装到李沐的d2l包里了
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    torch.save(net.state_dict(), 'vgg16.params')


    