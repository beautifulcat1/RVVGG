import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils import data

class MyData(Dataset):
    def __init__(self, root_dir, label_dir, dirction, type, transform = None):
        '''
        :param: root_dir:源文件所在的地址
        :param: label_dir:类别文件所在的地址
        :param: dirction:对类别的文件名进行映射为数字，方便分类存储
        :param: type: 训练集还是测试集
        :param: transform:将加载的数据进行变换
        '''
        self.transform = transform
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.dir = dirction
        self.img_path = os.path.join(self.root_dir,self.label_dir,type)
        self.img_name_list = os.listdir(self.img_path)
    def __getitem__(self, index):
        '''
        :param: index:类自动调用，遍历__len__()返回的长度
        :param: self.img_path:存放图片的文件的路径
        :param: self.img_name_list:存图片的路径中图片的名字列表
        :param: img_name:图片名字列表中某个图片的名字
        :param: img_item_path:某个图片地址 + 图片的名字
        :return 返回一个图片的矩阵和类别的数字
        '''
        img_name = self.img_name_list[index]
        img_item_path = os.path.join(self.img_path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        label = self.dir[label]
        if self.transform == None:
            return img, label
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_name_list)

def load_my_dataset(root_dir, batch_size, num_class, shuffle = True, resize=None): 
    '''
    :param root_dir:数据集所在的路径
    :param batch_size:批次大小
    :param num_class:加载类的数量
    :param shuffle:是否打乱
    :param resize:重新裁剪
    :return 返回两个数据集，分别是训练集和测试集
    '''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    
    flist = os.listdir(root_dir)
    dir = {}
    for i,name in enumerate(flist):
        dir[name] = i
    it = iter(flist)
    label = next(it)
    dataset_train =  MyData(root_dir, label, dir, 'train', trans)
    dataset_test = MyData(root_dir, label, dir, 'test', trans)
    for i in range(num_class-1):
        label = next(it)
        dataset_train = dataset_train + MyData(root_dir, label, dir, 'train', trans)
        dataset_test = dataset_test + MyData(root_dir, label, dir, 'test', trans)
    return (data.DataLoader(dataset_train, batch_size, shuffle), data.DataLoader(dataset_test, batch_size, shuffle))


if __name__ == '__main__':
    root_dir = "D:\\modelnet40\\modelnet40(1)"
    train_iter, test_iter = load_my_dataset(root_dir, 225, 10, shuffle = False, resize=(224,224))
    it = iter(train_iter)
    X,y = next(it)
    print(X.shape)
    # print(y)
    # img = X[1].numpy()
    # img = img.swapaxes(0, 1)
    # img = img.swapaxes(1, 2)
    # plt.imshow(img)
    # plt.show()

