import numpy as np
import pandas as pd
from keras import applications
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam
from random import random, randint
from scipy.ndimage import zoom
from skimage.transform import rotate

def getModel(size):
    model = Sequential()
    model.add(Dense(128, input_shape=(size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Nadam(0.001), metrics=['accuracy'])
    return model

def getImages(pd_set, h, w):
    band_1 = np.array([np.array(band).astype(np.float32).reshape(h, w) for band in pd_set["band_1"]])
    band_1_scale = 0.5 + ((band_1 - band_1.mean()) / (band_1.max() - band_1.min()))
    band_2 = np.array([np.array(band).astype(np.float32).reshape(h, w) for band in pd_set["band_2"]])
    band_2_scale = 0.5 + ((band_2 - band_2.mean()) / (band_2.max() - band_2.min()))
    band_3 = (band_1 + band_2) / 2
    band_3_scale = 0.5 + ((band_3 - band_3.mean()) / (band_3.max() - band_3.min()))
    img = np.concatenate([band_1_scale[:, :, :, np.newaxis], band_2_scale[:, :, :, np.newaxis], band_3_scale[:, :, :, np.newaxis]], axis=-1)
    return img

def preprocessAngle(pd_set):
    angle = pd_set["inc_angle"]
    angle = angle.as_matrix(columns=None)
    angle[angle == 'na'] = 0
    angle = (angle - angle.mean()) / (angle.max() - angle.min())
    return angle

def colorDistortion(X):
    for i in range(X.shape[0]):
        channel = randint(0,2)
        X[i,:,:,channel] = X[i,:,:,channel] + (random() * 0.1)
    return X

def mirroring(X):
    mirror_images = []
    for m in range(X.shape[0]):
        band_1 = X[m, :, :, 0]
        band_2 = X[m, :, :, 1]
        band_3 = X[m, :, :, 2]
        band_1_lr = np.fliplr(band_1)
        band_2_lr = np.fliplr(band_2)
        band_3_lr = np.fliplr(band_3)
        mirror_images.append(np.dstack((band_1_lr, band_2_lr, band_3_lr)))
    return mirror_images

def flipUpDown(X):
    flip_images = []
    for m in range(X.shape[0]):
        band_1 = X[m, :, :, 0]
        band_2 = X[m, :, :, 1]
        band_3 = X[m, :, :, 2]
        band_1_ud = np.flipud(band_1)
        band_2_ud = np.flipud(band_2)
        band_3_ud = np.flipud(band_3)
        flip_images.append(np.dstack((band_1_ud, band_2_ud, band_3_ud)))
    return flip_images

def randomRotate(X, h, w):
    for i in range(X.shape[0]):
        angle = randint(0,180)
        img = X[i,:,:,:]
        f0 = rotate(img[:,:,0], angle, resize=True)
        f1 = rotate(img[:,:,1], angle, resize=True)
        f2 = rotate(img[:,:,2], angle, resize=True)
        f = np.dstack((f0, f1, f2))
        X[i,:,:,:] = f[0:h,0:w,:]
    return X

def imageZoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    if zoom_factor < 1:
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
    elif zoom_factor > 1:
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = img
    return out

def randomZoom(X):
    for i in range(X.shape[0]):
        zoom_factor = random() + 0.5
        X[i,:,:,:] = imageZoom(X[i,:,:,:], zoom_factor)
    return X

def whiteNoise(X):
    r = 0.1*np.random.rand(X.shape[0],X.shape[1],X.shape[2],X.shape[3])
    X = X + r
    return X

def randomCropping(X, images_height, images_width):
    for i in range( X.shape[0]):
        new_img = np.zeros((images_height, images_width, 3))
        s = np.random.randint(10, size=4)
        new_img[s[0]:(images_height-s[1]),s[2]:(images_width-s[3]),:] = X[i,s[0]:(images_height-s[1]),s[2]:(images_width-s[3]),:]
        X[i,:,:,:] = new_img
    return X

def dataAugmentation(X, y1, y2, h, w):
    replications = 8
    y1 = np.tile(y1, (replications,1))
    y2 = np.tile(y2, (replications,1))
    X1 = colorDistortion(X)
    X2 = whiteNoise(X)
    X3 = mirroring(X)
    X4 = flipUpDown(X)
    X5 = randomRotate(X, h, w)
    X6 = randomZoom(X)
    X7 = randomCropping(X, h, w)
    X = np.concatenate([X, X1, X2, X3, X4, X5, X6, X7], axis=0)
    return [X,y1,y2]

print('Iceberg script started')
print('Loading data...')
images_width = 75
images_height = 75
file_path = ".model_weights.hdf5"
train = pd.read_json("data/processed/train.json")
test = pd.read_json("data/processed/test.json")
print('Done')

print('Data preprocessing...')
images_train = getImages(train, images_height, images_width)
images_test = getImages(test, images_height, images_width)
y_train = np.reshape(train["is_iceberg"], [images_train.shape[0],1])
angle_train = np.reshape(preprocessAngle(train), [images_train.shape[0],1])
angle_test = np.reshape(preprocessAngle(test), [images_test.shape[0],1])
#[images_train, y_train, angle_train] = dataAugmentation(images_train, y_train, angle_train, images_height, images_width)
print('Done')

print('Extract features using VGG16...')
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
base_features_train = base_model.predict(images_train)
base_features_test = base_model.predict(images_test)
print('Done')

print('Train a new model...')
base_features_train = np.reshape(base_features_train, [images_train.shape[0], base_features_train.shape[1] * base_features_train.shape[2] * base_features_train.shape[3]])
size = base_features_train.shape[1]+1
X_train = np.zeros((images_train.shape[0], size))
X_train[:,0] = np.reshape(angle_train,[images_train.shape[0]])
base_features_train = (base_features_train - base_features_train.mean()) / (base_features_train.max() - base_features_train.min())
X_train[:,1:] = base_features_train
base_features_test = np.reshape(base_features_test, [images_test.shape[0], base_features_test.shape[1] * base_features_test.shape[2] * base_features_test.shape[3]])
X_test = np.zeros((images_test.shape[0], size))
X_test[:,0] = np.reshape(angle_test, [images_test.shape[0]])
base_features_test = (base_features_test - base_features_test.mean()) / (base_features_test.max() - base_features_test.min())
X_test[:,1:] = base_features_test
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, random_state=1, train_size=0.75)
model = getModel(size)
model.fit(X_train, y_train, batch_size=128, epochs=100,validation_data=(X_cv, y_cv),callbacks=[ModelCheckpoint(filepath=file_path,save_best_only=True, mode='min', monitor='loss')])
model = load_model(file_path)
print('Done')


print('Model evaluate...')
score = model.evaluate(X_cv, y_cv)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
print('Done')

print('Start predictions...')
test_predictions = model.predict(X_test)
print('Done')
print('Start submission...')
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=test_predictions.reshape((test_predictions.shape[0]))
submission.to_csv('sub.csv', index=False)
print('Done')



