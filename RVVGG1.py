# 对数据进行预处理，进行resize并保存到指定路径
import cv2
import os
import numpy as np
from PIL import Image
 
def pic_compress_png(image_path,new_image_path):
    '''
    将图片压缩成png格式
    :param image_path:  原始文件路径
    :param new_image_path:  保存文件路径
    :return:
    '''
    files = os.listdir(image_path)  # 获取当前路径下的所有文件名字
    files = np.sort(files)         #按名称排序
    i = 0
    for f in files:
        imgpath = os.path.join(image_path, f)    #路径+文件名字
        img = cv2.imread(imgpath)   #读取图片
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        dirpath = new_image_path       #压缩后存储路径
        file_name, file_extend = os.path.splitext(f)   #将文件名的，名字和后缀进行分割
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.png')  #文件最终保存的路径及名字（名字和压缩前的名字一致），
        print(os.path.join(dirpath,"1.png"))  #打印压缩缓存文件路径
        shrink = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA) #对图像的大小进行resize   4864 *1024
        cv2.imwrite(os.path.join(dirpath,"1.png"), shrink, [cv2.IMWRITE_PNG_COMPRESSION, 1]) #对图像进行压缩 【cv2.IMWRITE_PNG_COMPRESSION, 1】
                                                                                            #v2.IMWRITE_PNG_COMPRESSION  压缩品质 0-10 ，数字越小压缩比越小
        img1 = Image.open(os.path.join(dirpath,"1.png"))    #打开压缩后的缓冲文件
        img1.save(dst,quality=70)                          #二次压缩，并保存位原始文件的文件名
        os.remove(os.path.join(dirpath,"1.png"))           #删除缓存文件
 

class compress():

    def __init__(self, ori_data_path, com_data_path):
        self.ori_data_path = ori_data_path
        self.com_data_path = com_data_path
        self.list1 = []
        self.list2 = []
    def pre(self, ori_data_path, com_data_path):
        '''
        递归先序遍历文件夹，若是有png或jpg则将路径添加到list中
        :param ori_data_path:要压缩的文件夹路径
        :param com_data_path:压缩后的文件夹路径
        :return 
        ''' 
        ori_data_list = os.listdir(ori_data_path)
        for name in ori_data_list:
            if (name[-4:] == '.jpg'  or name[-4:] == '.png'):#是照片则将文件路径直接添加到list中
                if  len(self.list1) == 0:#为空直接添加
                    self.list1.append(ori_data_path)
                    self.list2.append(com_data_path)
                else:
                    if not self.list1[-1] == ori_data_path:#若列表已经有啦照片的路径则不必重复添加
                        self.list1.append(ori_data_path)
                        self.list2.append(com_data_path)
            else:#是一个文件夹则进行递归
                self.pre(os.path.join(ori_data_path, name),os.path.join(com_data_path, name))
    def get_path(self):
        self.pre(self.ori_data_path, self.com_data_path)
        return self.list1,self.list2

if __name__ == '__main__':
    path1, path2 = r"D:\modelnet40\modelnet40(1)", r"D:\test"
    c = compress(path1,path2)
    
    path1, path2 = c.get_path()
    for path1, path2 in zip(path1,path2):
        os.makedirs(path2)
        pic_compress_png(path1,path2)




