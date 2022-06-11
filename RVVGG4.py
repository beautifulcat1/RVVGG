# 搜索过程
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
from matplotlib import pyplot as plt

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



# 获取出现最多的下标
def find_max_times_index(in_list):
    '''
    一个模型投影的18张图，通过vgg算得18张图的所在类，通过计算其中出现次数最多的类别，代表模型所在的类别
    :param: in_list:18张图所在的类别的数组
    :return 返回类别
    '''
    list =[0] * len(in_list)
    for i in in_list:
        list[i] += 1
    max_point = 0
    for i in range(len(list)):
        if list[i] > list[max_point]:
            max_point = i
    return max_point 
# 计算两张图的特征向量的距离
def distance_of_two_feature(index_feature_of_photo, search_feature_of_photo):
    '''
    计算两个向量的距离
    :param: index_feature_of_photo: 索引库中某个照片的特征向量
    :param: search_feature_of_photo: 要检索的照片的特征向量
    :return 返回两个向量的距离
    '''
    distance = 0.0
    for i in range(len(index_feature_of_photo)):
        distance += (index_feature_of_photo[i] - search_feature_of_photo[i])**2 
    return distance
# 查询类中模型的索引 search_label:第几类， index：类中第几个模型
def find_index_of_class(index_features, search_features, search_label_list):
    '''
    根据输入模型的投影图片返回模型所在的类别，所在类别的模型索引（即第几类第几个模型）
    :param: index_features:索引库的特征
    :param: search_features:要检索模型的特征矩阵
    :param: search_label_list:投影出18张图所在的类别的数组
    :return: 返回所在类别，类中的第几个模型
    '''
    search_label = find_max_times_index(search_label_list)
    index_features_of_class = index_features[search_label]
    max_index_of_class_list = [0] * len(index_features_of_class)

    # i是输入特征矩阵的一个特征向量，特征向量分别与索引库中的模型的特征向量计算距离
    # 比如第三个输入特征向量，分别与索引库中对应类的第一个模型的第三个特征向量，第二个模型的第三个特征向量，第三个。。。计算距离
    # 每个输入特征向量求得一个最小的距离并记下是第几个模型，最后查看哪个模型的得票最多，则与其最贴近
    for i in range(len(search_features)):
        min_distance_index = 0
        min_distance = distance_of_two_feature(index_features_of_class[min_distance_index][i], search_features[i])
        for j in range(len(index_features_of_class)):
            distance = distance_of_two_feature(index_features_of_class[j][i], search_features[i])
            if min_distance > distance:
                min_distance = distance
                min_distance_index = j
        max_index_of_class_list[min_distance_index] += 1

    # 查询出现模型中投票最多的索引
    index = 0
    for i in range(len(max_index_of_class_list)):
        if max_index_of_class_list[i] > max_index_of_class_list[index]:
            index = i

    return search_label, index


if __name__== '__main__':
    conv_arch = ((1, 64), (1, 128), (1, 256), (1, 512), (1, 512))
    net = vgg(conv_arch)
    net.load_state_dict(torch.load('vgg16.params'))
    index_features = np.load('features.npy', allow_pickle=True)

    root_dir = r"D:\test"
    slips_num, num_class = 18, 10 
    train_iter, test_iter = d2l.load_my_dataset(root_dir, slips_num, num_class, shuffle = False, resize=(224,224))


    it = iter(test_iter)
    X, y = next(it)

    out_features = net(X)#输出一个与18张图组成的18*10的特征矩阵

    out_features = out_features.detach()#去掉梯度
    out_label_list = out_features.argmax(axis = 1)#求得每张图所得的最大值，即每张图的类别

    out_features = out_features.tolist()#将torch的张量转换为python的list
    out_label_list = out_label_list.tolist()


    type, index = find_index_of_class(index_features, out_features, out_label_list)
    print(index)


    # img = X[0].numpy()
    # img = img.swapaxes(0, 1)
    # img = img.swapaxes(1, 2)
    # plt.imshow(img)
    # plt.show()