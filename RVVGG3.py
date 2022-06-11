# 建立索引过程
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

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


if __name__ == '__main__':
#   初始化
    conv_arch = ((1, 64), (1, 128), (1, 256), (1, 512), (1, 512))
    net = vgg(conv_arch)
    net.load_state_dict(torch.load('vgg16.params'))
    root_dir = r"D:\test"
    slips_num, num_class = 18, 10 #一个模型投影出图片的数量，类别的数量
    train_iter, test_iter = d2l.load_my_dataset(root_dir, slips_num, num_class, shuffle = False, resize=(224,224))#不能打乱顺序，因为一个模型的投影是连续的

#   建立索引的过程如下
    feature_matrix_all = [] 
    it = iter(train_iter)
    X, y = next(it)
    model_total_nums = 0#模型的总数
    for i in range(num_class):#对每个类分别建立索引
        feature_matrix_all.append([])
        print(f"种类{i}正在建立索引。。。")    
        j = 0
        #y[0]用以判断取出的数据是否是一个类的，退出下边的循环，第二个条件防止循环一直迭代到迭代器没有数据
        while y[0] == i and (model_total_nums < (len(train_iter) - 1)) :
            print(f"模型{j}已完成索引建立")
            feature_matrix = net(X)
            feature_matrix_all[i].append(feature_matrix.tolist())
            j += 1
            model_total_nums += 1
            X, y = next(it)
            
    feature_matrix_all = np.array(feature_matrix_all)#feature_matrix_all一个4维的向量，第一维为类别，第二维为类中模型，第三维为某个模型，第四维为某个模型的某个投影照片的特征向量
    np.save('features',feature_matrix_all)
