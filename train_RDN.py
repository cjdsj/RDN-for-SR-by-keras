import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from data_processing import get_train_data
from model import RDN, L1_loss


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
train_folders = ['/content/drive/MyDrive/Colaboratory/跨平台超分辨率/SR_train&test/SR_training_datasets/BSDS200',
                 '/content/drive/MyDrive/Colaboratory/跨平台超分辨率/SR_train&test/SR_training_datasets/General100',
                 '/content/drive/MyDrive/Colaboratory/跨平台超分辨率/SR_train&test/SR_training_datasets/T91']
val_folders = ['/content/drive/MyDrive/Colaboratory/跨平台超分辨率/SR_train&test/SR_testing_datasets/Set5']
model_save_path = '/content/drive/MyDrive/Colaboratory/跨平台超分辨率/models/RDN.h5'
Imglist = ['RGB', 'GRAY', 'YCrCb']
print('Have got parameters, for ' + Imglist[Imgflag] + ' images.')


''' Get training and validation data '''
y_train = get_train_data(train_folders, size, scale, Rflag=0, Imgflag=Imgflag, aug_num=aug_num)
x_train = get_train_data(train_folders, size, scale, Rflag=1, Imgflag=Imgflag, aug_num=aug_num)
y_val = get_train_data(val_folders, size, scale, Rflag=0, Imgflag=Imgflag)
x_val = get_train_data(val_folders, size, scale, Rflag=1, Imgflag=Imgflag)
if Imgflag != 0:  # Have to add another dimension for channel if images only have one channel
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], y_train.shape[2], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], y_val.shape[2], 1))
y_train, x_train = y_train / 255.0, x_train / 255.0  # Standardization
y_val, x_val = y_val / 255.0, x_val / 255.0
print('\n')
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


''' Build the model '''
model = RDN(num_G=num_G, channels=channels, scale=scale)
model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss=L1_loss)


''' Train the model and save it'''
callback_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.707, patience=2, verbose=1)]
model.fit(x_train, y_train, batch_size=16, epochs=100, validation_data=(x_val, y_val), callbacks=callback_list)
model.save(model_save_path)