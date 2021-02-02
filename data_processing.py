import cv2
import os
import numpy as np
import math
from tqdm import tqdm


def delcrust(HR_img, scale, Imgflag, Rflag=0):
    HR_size = HR_img.shape
    rem0 = HR_size[0] % scale
    rem1 = HR_size[1] % scale
    if Imgflag == 0:
        HR_img = HR_img[:HR_size[0] - rem0, :HR_size[1] - rem1, :]  # Cut edges that cannot be divisible by scale
    else:
        HR_img = HR_img[:HR_size[0] - rem0, :HR_size[1] - rem1]
    ''' Rflag=0，get HR；Rflag=1，get LR；Rflag=2，get ILR'''
    if Rflag == 0:
        return HR_img
    if Rflag == 1 or Rflag == 2:
        HR_size = HR_img.shape
        HR_size = (HR_size[1], HR_size[0])
        LR_size = (int(HR_size[0] / scale), int(HR_size[1] / scale))
        LR_img = cv2.resize(HR_img, LR_size, interpolation=cv2.INTER_LINEAR)
        if Rflag == 1:
            return LR_img
        else:
            ILR_img = cv2.resize(LR_img, HR_size, interpolation=cv2.INTER_LINEAR)  # Restore to HR's size
            return ILR_img


def dataaug(img, size, flag):
    if flag == 0:
        img = cv2.flip(img, 0)  # Flip vertical
    elif flag == 1:
        img = cv2.flip(img, 1)  # Flip horizontal
    elif flag == 2:
        center = (size // 2, size // 2)
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 90, 1), (size, size))  # Rotate 90 degrees
    elif flag == 3:
        center = (size // 2, size // 2)
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 270, 1), (size, size))  # Rotate 270 degrees
    return img


def splitdata(img, size, Imgflag, aug_num=0):
    imglist = []
    length_num = img.shape[0] // size  # Calculate for how many blocks can be divided into
    width_num = img.shape[1] // size
    for i in range(0, length_num):
        for j in range(0, width_num):
            if Imgflag == 0:
                img_piece = img[(0 + i * size):(size + i * size), (0 + j * size):(size + j * size), :]
            else:
                img_piece = img[(0 + i * size):(size + i * size), (0 + j * size):(size + j * size)]
            imglist.append(img_piece)
            for k in range(0, aug_num):  # The maximum value is 4
                imglist.append(dataaug(img_piece, size, k))
    imglist = np.array(imglist)
    return imglist


def get_train_data(folder_name_list, size, scale, Rflag=0, Imgflag=0, sizeflag=0, aug_num=0):
    firstflag = 0
    ''' The way of reading images. 0: RGB, 1: GRAY, 2: Y channel from Y_CrCb form '''
    if Imgflag == 0 or Imgflag == 2:
        Imgform = cv2.IMREAD_COLOR
    else:
        Imgform = cv2.IMREAD_GRAYSCALE
    for folder_name in folder_name_list:
        for img_name in tqdm(os.listdir(folder_name)):
            file_name = folder_name + '/' + img_name
            img = cv2.imread(file_name, Imgform)  # Read images
            if Imgflag == 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  # Read images
                img = img[:, :, 0]  # Get Y channel
            ''' Rflag=0，get HR；Rflag=1，get LR；Rflag=2，get ILR'''
            img = delcrust(img, scale, Imgflag, Rflag)  # Cut edges that cannot be divisible by scale
            if (Rflag == 0 and sizeflag == 0) or (Rflag != 0 and sizeflag == 1):
                # For HR, splitsize is size*scale，others are size. But it could be changed by changing sizeflag
                splitsize = size * scale
            else:
                splitsize = size
            sub_imglist = splitdata(img, splitsize, Imgflag, aug_num)
            if firstflag == 0:
                imglist = sub_imglist.copy()  # Generate imglist ate the begining
                firstflag += 1
            else:
                imglist = np.append(imglist, sub_imglist, axis=0)
    return imglist


def get_test_data(file_name, size, scale, Rflag=0, Imgflag=0, sizeflag=0):
    ''' The way of reading images. 0: RGB, 1: GRAY, 2: Y channel from Y_CrCb form '''
    if Imgflag == 0 or Imgflag == 2:
        Imgform = cv2.IMREAD_COLOR
    else:
        Imgform = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(file_name, Imgform)  # Read images
    if Imgflag == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  # Read images
    ''' Rflag=0，get HR；Rflag=1，get LR；Rflag=2，get ILR '''
    img = delcrust(img, scale, Imgflag, Rflag)  # Cut edges that cannot be divisible by scale
    if Imgflag == 2:  # For Y_CrCb form, we only need to learn how to recover Y channel
        img_res = img[:, :, -2:]  # Cr and Cb channels
        img = img[:, :, 0]  # Y channel
    else:
        img_res = None
    if (Rflag == 0 and sizeflag == 0) or (Rflag != 0 and sizeflag == 1):
        splitsize = size * scale
    else:
        splitsize = size
    imglist = splitdata(img, splitsize, Imgflag)
    return imglist, img, img_res


def imgappend(img_pieces, length_num, width_num):
    num = 0
    for i in range(0, length_num):
        for j in range(0, width_num):
            if j == 0:
                width_array = img_pieces[num, :, :]
            else:
                width_array = np.append(width_array, img_pieces[num, :, :, :], axis=1)  # Horizontal splicing
            num += 1
        if i == 0:
            length_array = width_array.copy()
        else:
            length_array = np.append(length_array, width_array, axis=0)  # Vertical splicing
    return length_array


def getYimg(ILR_img, pred_img, HR_img, LR_res, HR_res):
    HR_img = np.reshape(HR_img, (HR_img.shape[0], HR_img.shape[1], 1))
    ILR_img = np.reshape(ILR_img, (ILR_img.shape[0], ILR_img.shape[1], 1))
    pred_img = np.reshape(pred_img, (ILR_img.shape[0], ILR_img.shape[1], 1))
    ILR_res = cv2.resize(LR_res, (LR_res.shape[1] * scale, LR_res.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
    HR_res = HR_res[:HR_img.shape[0], :HR_img.shape[1], :]
    ILR_res = ILR_res[:ILR_img.shape[0], :ILR_img.shape[1], :]
    HR_img = np.append(HR_img, HR_res, axis=2)
    ILR_img = np.append(ILR_img, ILR_res, axis=2)
    pred_img = np.append(pred_img, ILR_res, axis=2)
    ILR_img = cv2.cvtColor(ILR_img, cv2.COLOR_YCR_CB2BGR)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_YCR_CB2BGR)
    HR_img = cv2.cvtColor(HR_img, cv2.COLOR_YCR_CB2BGR)
    return ILR_img, pred_img, HR_img


def psnr(y_true, y_pred):
    y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2YCR_CB)
    y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2YCR_CB)
    y_true = y_true[:, :, 0]
    y_pred = y_pred[:, :, 0]
    mse = np.mean((y_true / 1.0 - y_pred / 1.0) ** 2)
    return 10 * math.log10(255.0 ** 2 / mse)