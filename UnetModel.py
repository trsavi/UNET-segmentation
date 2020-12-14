# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K

random.seed(10)


# Set parameters
im_width = 256
im_height = 256
path_train = '..\Training'
path_test =  '..\Test'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "\Images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/Images/' + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (256, 256, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/Masks/' + id_, color_mode = "grayscale"))
            mask = resize(mask, (256, 256, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X


    
X, y = get_data(path_train, train=True)

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2018)


# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0],  interpolation='bilinear') #cmap='seismic',
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Rendgen')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Mask')



# =============================================================================
# # Plot loss and metrics
# =============================================================================

def plot(metric='accuracy'):
    
    if metric=='accuracy':
        pass
    else:
        
        metric = str(metric.__name__)
    metric_name=metric
    val_metric = 'val_' + metric
    metric = results.history[metric]
    val_metric = results.history[val_metric]
    
   
    loss = 'loss'
    val_loss = 'val_loss'
        
    epochs_range = range(len(results.history[loss]))
    #print(epochs_range)
    loss = results.history[loss]
    val_loss = results.history[val_loss]
    
    
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    
    plt.plot(epochs_range, metric, label='Training Metric')
    plt.plot(epochs_range, val_metric, label='Validation Metric')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Metric: '+ metric_name)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    #plt.ylim(0.4,2)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot2():
    fig, axs = plt.subplots(1, 2, figsize = (15, 4))

    training_loss = results.history['loss']
    validation_loss = results.history['val_loss']
    
    training_accuracy = results.history['iou_coef']
    validation_accuracy = results.history['val_iou_coef']
    
    epoch_count = range(1, len(training_loss) + 1)
    
    axs[0].plot(epoch_count, training_loss, 'r--')
    axs[0].plot(epoch_count, validation_loss, 'b-')
    axs[0].legend(['Training Loss', 'Validation Loss'])
    
    axs[1].plot(epoch_count, training_accuracy, 'r--')
    axs[1].plot(epoch_count, validation_accuracy, 'b-')
    axs[1].legend(['Training IoU', 'Validation IoU'])
    plt.show()

# =============================================================================
# # Define metrics and loss for segmentation problem
# =============================================================================
def jaccard_loss(y_true, y_pred, smooth=1):
    """
    Arguments:
        y_true : Matrix containing one-hot encoded class labels 
                 with the last axis being the number of classes.
        y_pred : Matrix with same dimensions as y_true.
        smooth : smoothing factor for loss function.
    """

    intersection = tf.reduce_sum( y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)
    
    return (1 - jac) * smooth


def f1_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    f1 = (2 * intersection + smooth) / (denominator + smooth)
    
    return (1 - f1) * smooth

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

# without smooth
def dice_coef_v2(y_true, y_pred):
    y_true_f = tf.keras.flatten(y_true)
    y_pred_f = tf.keras.flatten(y_pred)
    intersection = tf.keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.sum(y_true_f) + tf.keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# =============================================================================
# # Define the UNET
# =============================================================================

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    #p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    #p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    #p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    #p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    #u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    #u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    #u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True)


# =============================================================================
# ## Binary crossentropy loss with pixel accuracy
# =============================================================================
#model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[metric])

# =============================================================================
# ## Jaccard loss with dice coefficient and binary accuracy
# =============================================================================
# model.compile(optimizer=Adam(lr=1e-3), loss=jaccard_loss, metrics=[dice_coef, 'binary_accuracy'])

model.compile(optimizer=Adam(lr=1e-3), loss=f1_loss, \
                  metrics=[iou_coef, 'binary_accuracy'])
model.summary() 
model_save = 'modelF1Loss.h5'
callbacks = [
        EarlyStopping(patience=5, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(model_save, verbose=1, save_best_only=True)
    ]
results = model.fit(X_train, y_train, batch_size=2, epochs=52, callbacks=callbacks, validation_data=(X_valid, y_valid))
plot2()


# =============================================================================
# ## Predictions 
# =============================================================================

#from PIL import Image
X_test, y_test = get_data(path_test, train=True)

prediction = model.predict(X_test)

fig, ax = plt.subplots(1, 3, figsize=(20, 15))

ax[0].imshow(X_test.squeeze(), interpolation='bilinear') 

ax[0].set_title('Ground Truth (Image)')

ax[1].imshow(y_test.squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('True Mask')

ax[2].imshow(prediction.squeeze(), interpolation='bilinear', cmap='gray')
ax[2].set_title('Predicted')






