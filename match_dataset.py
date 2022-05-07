import numpy
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
from torchvision import transforms
import torch


def PIL2array(_img: Image.Image, input_pixel) -> numpy.ndarray:
    """Convert PIL image type to numpy 2D array"""
    return numpy.array(_img.getdata(), dtype=numpy.uint8).reshape(input_pixel, input_pixel)

class MatchDataset(Dataset):
    def __init__(self, root_a, root_b, input_pixel):
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])
        self.root_a = os.listdir(root_a)
        self.root_b = os.listdir(root_b)
        self.transform = transform
        self.imagePairs = []
        self.matchLabels = []
        self.input_pixel = input_pixel

        if self.root_a != self.root_b:
            print("Data Error!!!")
            return

        for file in self.root_a:
            # 分别从A文件夹中按顺序抽取imgX.jpg,组成匹配样本
            imgLeft = Image.open(root_a + file)
            imgRight = Image.open(root_b + file)
            imgLeft_tmp = imgLeft.resize((self.input_pixel, self.input_pixel))
            imgLeft_tmp = ImageOps.grayscale(imgLeft_tmp)
#            imgLeft_final = keras.preprocessing.image.img_to_array(imgLeft_tmp)
            imgLeft_final = PIL2array(imgLeft_tmp, self.input_pixel)
            imgRight_tmp = imgRight.resize((self.input_pixel, self.input_pixel))
            imgRight_tmp = ImageOps.grayscale(imgRight_tmp)
#            imgRight_final = keras.preprocessing.image.img_to_array(imgRight_tmp)
            imgRight_final = PIL2array(imgRight_tmp, self.input_pixel)

            self.imagePairs.append([imgLeft_final, imgRight_final])
            self.matchLabels.append(1)

            # numpy.random.choice(pathTempB)的意思是,对于tempA文件夹中imgX.jpg,从tempB文件夹中随机抽取一个构建负样本
            # 从概率上讲,此时的imgX.jpg与imgY.jpg是不匹配的
            is_same = True
            while(is_same):
                tmp_str = numpy.random.choice(self.root_b)
                if tmp_str is not file:
                    is_same = False
            imgRightFalse = Image.open(root_b + tmp_str)
            imgRightFalse_tmp = imgRightFalse.resize((self.input_pixel, self.input_pixel))
            imgRightFalse_tmp = ImageOps.grayscale(imgRightFalse_tmp)
#            imgRightFalse_final = keras.preprocessing.image.img_to_array(imgRightFalse_tmp)
            imgRightFalse_final = PIL2array(imgRightFalse_tmp, self.input_pixel)
            self.imagePairs.append([imgLeft_final, imgRightFalse_final])
            self.matchLabels.append(0)

            # is_same = True
            # while (is_same):
            #     tmp_str = numpy.random.choice(self.root_b)
            #     if tmp_str is not file:
            #         is_same = False
            # tmp_str = numpy.random.choice(self.root_b)
            # imgRightFalse = Image.open(root_b + tmp_str)
            # imgRightFalse_tmp = imgRightFalse.resize((64, 64))
            # imgRightFalse_tmp = ImageOps.grayscale(imgRightFalse_tmp)
            # #            imgRightFalse_final = keras.preprocessing.image.img_to_array(imgRightFalse_tmp)
            # imgRightFalse_final = PIL2array(imgRightFalse_tmp)
            # self.imagePairs.append([imgLeft_final, imgRightFalse_final])
            # self.matchLabels.append(0)
            #
            # is_same = True
            # while (is_same):
            #     tmp_str = numpy.random.choice(self.root_b)
            #     if tmp_str is not file:
            #         is_same = False
            # tmp_str = numpy.random.choice(self.root_b)
            # imgRightFalse = Image.open(root_b + tmp_str)
            # imgRightFalse_tmp = imgRightFalse.resize((64, 64))
            # imgRightFalse_tmp = ImageOps.grayscale(imgRightFalse_tmp)
            # #            imgRightFalse_final = keras.preprocessing.image.img_to_array(imgRightFalse_tmp)
            # imgRightFalse_final = PIL2array(imgRightFalse_tmp)
            # self.imagePairs.append([imgLeft_final, imgRightFalse_final])
            # self.matchLabels.append(0)

    def __len__(self):
        return len(self.matchLabels)

    def __getitem__(self, idx):
        imgPairData = self.imagePairs[idx]
#        print("len of imgPairData = "+str(len(imgPairData)) + "\n")
        imgData1 = imgPairData[0]
        imgData2 = imgPairData[1]
        m = self.matchLabels[idx]
#        print(m)
        if self.transform is not None:
            imgData1 = self.transform(imgData1)
            imgData2 = self.transform(imgData2)
        return imgData1, imgData2, torch.tensor(m)
