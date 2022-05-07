import torch
import cv2
import numpy as np
import os
from model import ClassiFilerNet
from PIL import Image, ImageOps
import numpy
from torchvision import transforms

# 分成a目录和b目录，match的pair以相同文件名如x1.jpg分别放在a和b目录下。本测试会遍历a目录
# 下的所有文件，和b目录下的所有文件进行match测试。

imgDir_a = "./test/a/"
imgDir_b = "./test/b/"


def PIL2array(_img: Image.Image) -> numpy.ndarray:
    """Convert PIL image type to numpy 2D array"""
    return numpy.array(_img.getdata(), dtype=numpy.uint8).reshape(64, 64)


if __name__ == "__main__":
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    model = ClassiFilerNet("alexNet")
    pthFile = "./good_model/model_best.pth.tar"
    print("=> loading checkpoint '{}'".format(pthFile))
    checkpoint = torch.load(pthFile)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(pthFile, checkpoint['epoch']))
    model.eval()

    results = []
    root_a = os.listdir(imgDir_a)
    root_b = os.listdir(imgDir_b)
    for file_a in root_a:
        imgLeft = Image.open(imgDir_a + file_a)
        imgLeft_tmp = imgLeft.resize((64, 64))
        imgLeft_tmp = ImageOps.grayscale(imgLeft_tmp)
        imgLeft_final = PIL2array(imgLeft_tmp)
        for file_b in root_b:
            imgRight = Image.open(imgDir_b + file_b)
            imgRight_tmp = imgRight.resize((64, 64))
            imgRight_tmp = ImageOps.grayscale(imgRight_tmp)
            imgRight_final = PIL2array(imgRight_tmp)

            img1 = torch.from_numpy(imgLeft_final).to(torch.float32).reshape(1, 1, 64, 64)
            img2 = torch.from_numpy(imgRight_final).to(torch.float32).reshape(1, 1, 64, 64)
            result = model((img1, img2))[0]
            print(file_a + " vs. " + file_b + " result is " + str(result[1]))

    #
    # for img in imgList:
    #     img = torch.from_numpy((img-128)/160).to(torch.float32)
    #     out = model((template, img))[0]
    #     print(out)
    #     results.append(out[1] > 0.55)

# print(results)
#
# # target patches
# imgList = np.array(imgList).reshape(1, 1, 1, 64, 64)
# # template patch
# template = cv2.imread("./test/template.png", 0)
# template = cv2.resize(template, (64, 64))
# template = torch.from_numpy((template-128)/160).to(torch.float32).reshape(1, 1, 64, 64)
# print(template.shape)
