import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from data_processing import get_test_data, imgappend, getYimg, psnr


''' Set parameters '''
scale = 2
size = 32  # Input size
aug_num = 4  # Number of image augmentations, the maximum value is 4
num_G = 32  # Number of convolution kernels
lr = 1e-4  # Learning rate
Imgflag = 2  # The way of reading images. 0: RGB, 1: GRAY, 2: Y channel from Y_CrCb form
if Imgflag == 0:
    channels = 3
else:
    channels = 1
model_save_path = '/content/drive/MyDrive/Colaboratory/跨平台超分辨率/models/RDN.h5'
test_file = '/content/drive/MyDrive/Colaboratory/跨平台超分辨率/SR_train&test/SR_testing_datasets/Set5/butterfly.png'
Imglist = ['RGB', 'GRAY', 'YCrCb']
print('Have got parameters, for ' + Imglist[Imgflag] + ' images.')


'''Get test data '''
HR_test, HR_img, HR_res = get_test_data(test_file, size, scale, Rflag=0, Imgflag=Imgflag, sizeflag=0)
LR_test, LR_img, LR_res = get_test_data(test_file, size, scale, Rflag=1, Imgflag=Imgflag, sizeflag=0)
x_test, y_test = LR_test / 255.0, HR_test / 255.0
if Imgflag != 0:  # Have to add another dimension for channel if images only have one channel
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], y_test.shape[2], 1))
print(HR_img.shape, LR_img.shape)
if Imgflag == 2:  # Get Cr and Cb channels for Y_CrCb form
    print(HR_res.shape, LR_res.shape)
print(y_test.shape, x_test.shape)


''' Evaluate test image '''
model = load_model(model_save_path)
model.evaluate(x_test, y_test, batch_size=16)


''' Recover test image and show the result '''
y_pred = model.predict(x_test)

length_num = HR_img.shape[0] // (size * scale)
width_num = HR_img.shape[1] // (size * scale)
pred_img = imgappend(y_pred, length_num, width_num)
pred_img = (pred_img * 255).astype(np.uint8)
if Imgflag == 0:
    HR_img = HR_img[:pred_img.shape[0], :pred_img.shape[1], :pred_img.shape[2]]
    LR_img = LR_img[:pred_img.shape[0] // scale, :pred_img.shape[1] // scale, :pred_img.shape[2]]
else:
    HR_img = HR_img[:pred_img.shape[0], :pred_img.shape[1]]
    LR_img = LR_img[:pred_img.shape[0] // scale, :pred_img.shape[1] // scale]
    pred_img = np.reshape(pred_img, (pred_img.shape[0], pred_img.shape[1]))
ILR_img = cv2.resize(LR_img, (LR_img.shape[1] * scale, LR_img.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
if Imgflag == 2:
    ILR_img, pred_img, HR_img = getYimg(ILR_img, pred_img, HR_img, LR_res, HR_res)

if Imgflag != 1:  # Grayscale image can't calculate psnr
    cmap = None
    print('PSNR between ILR and HR:', psnr(HR_img, ILR_img))
    print('PSNR between output and HR:', psnr(HR_img, pred_img), '\n')

    b, g, r = cv2.split(pred_img)
    pred_img = cv2.merge([r, g, b])
    b, g, r = cv2.split(HR_img)
    HR_img = cv2.merge([r, g, b])
    b, g, r = cv2.split(ILR_img)
    ILR_img = cv2.merge([r, g, b])
else:
    cmap = plt.cm.gray

plt.figure(figsize=(15, 15))

plt.subplot(1, 3, 1)
plt.imshow(ILR_img, cmap)
plt.title("ILR")

plt.subplot(1, 3, 2)
plt.imshow(pred_img, cmap)
plt.title("Prediction")

plt.subplot(1, 3, 3)
plt.imshow(HR_img, cmap)
plt.title("HR")

plt.show()