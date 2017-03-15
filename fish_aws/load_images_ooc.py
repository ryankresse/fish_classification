import numpy as np
import cv2
import time
import os
import glob

def get_im_cv2(path, width, height):
    img = cv2.imread(path)
    resized = cv2.resize(img, (width, height), cv2.INTER_LINEAR)
    return resized

def load_next_batch(paths, width=80, height=45):
    X_train = []
    start_time = time.time()

    for fl in paths:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, width, height)
        X_train.append(img)
        
    
    X_train = np.array(X_train)
    X_train = X_train.astype('float32')
    
    #Normalize
    X_train = X_train / 255.0
    
   
    return X_train


def one_hot_encode(labels):
    labels_encoded = np.zeros((labels.shape[0],8))
    labels_encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return labels_encoded


def load_train(width=80, height=45, n_classes=8):
    X_train = []
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
            y_train.append(index)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    #one hot encode
    y_train_mod = np.zeros((y_train.shape[0],8))
    y_train_mod[np.arange(y_train.shape[0]), y_train ] = 1
    
    X_train = X_train.astype('float32')
    X_train = X_train / 255.0
    
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return np.array(X_train), y_train_mod

def load_test(width=80, height=45):
    path = os.path.join('input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))
    
    X_test = []
    X_test_id = []
    for i, fl in enumerate(files):
        if i % 20 == 0:
            print('loading {} of {}'.format(i, len(files)))
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, width, height)
        X_test.append(img)
        X_test_id.append(flbase)
        
    X_test = np.array(X_test)
    
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0

    return X_test, X_test_id

