import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam

def getModel():
    dropout = 0.2
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Nadam(0.001), metrics=['accuracy'])
    return model

def getImages(pd_set1, pd_set2, h, w):
    band_1_1 = np.array([np.array(band).astype(np.float32).reshape(h, w) for band in pd_set1["band_1"]])
    band_1_2 = np.array([np.array(band).astype(np.float32).reshape(h, w) for band in pd_set2["band_1"]])
    band_1 = np.concatenate([band_1_1, band_1_2], axis=0)
    band_1_1_scale = ((band_1_1 - band_1.mean()) / (band_1.max() - band_1.min()))
    band_1_2_scale = ((band_1_2 - band_1.mean()) / (band_1.max() - band_1.min()))
    band_2_1 = np.array([np.array(band).astype(np.float32).reshape(h, w) for band in pd_set1["band_2"]])
    band_2_2 = np.array([np.array(band).astype(np.float32).reshape(h, w) for band in pd_set2["band_2"]])
    band_2 = np.concatenate([band_2_1, band_2_2], axis=0)
    band_2_1_scale = ((band_2_1 - band_2.mean()) / (band_2.max() - band_2.min()))
    band_2_2_scale = ((band_2_2 - band_2.mean()) / (band_2.max() - band_2.min()))
    band_3_1 = (band_1_1_scale + band_2_1_scale) / 2
    band_3_2 = (band_1_2_scale + band_2_2_scale) / 2
    img1 = np.concatenate([band_1_1_scale[:, :, :, np.newaxis], band_2_1_scale[:, :, :, np.newaxis], band_3_1[:, :, :, np.newaxis]], axis=-1)
    img2 = np.concatenate([band_1_2_scale[:, :, :, np.newaxis], band_2_2_scale[:, :, :, np.newaxis], band_3_2[:, :, :, np.newaxis]], axis=-1)
    return [img1, img2]

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

def flipLeftRight(X):
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

def dataAugmentation(X, y, h, w):
    replications = 3
    y = np.tile(y, (replications,1))
    X1 = flipLeftRight(X)
    X2 = flipUpDown(X)
    X = np.concatenate([X, X1, X2], axis=0)
    return [X,y]

def shuffle(X, y):
    p = numpy.random.permutation(len(y))
    return [X[p,:,:,:], y[p]]

print('Iceberg script started')
print('Loading data...')
images_width = 75
images_height = 75
file_path = ".model_weights.hdf5"
train = pd.read_json("data/processed/train.json")
test = pd.read_json("data/processed/test.json")
print('Done')

print('Data preprocessing...')
[X_train, X_test] = getImages(train, test, images_height, images_width)
y_train = np.reshape(train["is_iceberg"], [X_train.shape[0],1])
[X_train, y_train] = shuffle(dataAugmentation(X_train, y_train, images_height, images_width))
print('Done')

print('Train a new model...')
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, random_state=1, train_size=0.75)
model = getModel()
model.fit(X_train, y_train, batch_size=64, epochs=15,validation_data=(X_cv, y_cv),callbacks=[ModelCheckpoint(filepath=file_path,save_best_only=True, mode='min', monitor='loss')])
model = load_model(file_path)
print('Done')

print('Model evaluate...')
score = model.evaluate(X_cv, y_cv)
print('Cross validation loss:', score[0])
print('Cross validation accuracy:', score[1])
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