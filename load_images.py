import numpy as np
import cv2
import time
import os
import glob

def get_im_cv2(path, width, height):
    img = cv2.imread(path)
    resized = cv2.resize(img, (width, height), cv2.INTER_LINEAR)
    return resized


def load_train(width=80, height=45, n_classes=8):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    #folders = ['SHARK', 'YFT']
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, width, height)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_train = X_train.astype('float32')
    X_train = X_train / 255.0
    
    y_zeros = np.zeros((y_train.shape[0], n_classes))
    y_zeros[np.arange(y_train.shape[0]), y_train] = 1

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return np.array(X_train), y_zeros, np.array(X_train_id)