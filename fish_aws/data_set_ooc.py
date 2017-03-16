from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import load_images_ooc
import os
import glob


class DataSet(object):

  def __init__(self,
               width,
               height,
               val_size=0.2):
    
    #normalize
    #images = np.multiply(images, 1.0 / 255.0)
    self._width = width
    self._height = height
    img_labels = pd.read_csv('train_name_labels.csv').values
    
    if val_size:
        X_train, X_val, y_train, y_val = train_test_split(img_labels[:,0], img_labels[:,1], test_size=val_size, stratify=img_labels[:,1])
    
    
    self._num_examples = X_train.shape[0]
    self._num_val_examples = X_val.shape[0]
    self._X_train = X_train
    self._y_train = y_train.astype(int)
    self._X_val = X_val
    self._y_val = y_val
    self._index_in_epoch = 0
    self._epochs_completed = 0
    self._index_in_val_epoch = 0
    self._val_epochs_completed = 0
    
    path = os.path.join('input', 'test_stg1', '*.jpg')
    self._test_files = np.array(glob.glob(path))
    self._index_in_test_set = 0
    self._num_test_examples = self._test_files.shape[0]
    
  @property
  def X_train(self):
    return self._X_train

  @property
  def y_train(self):
    return self._y_train

  @property
  def X_val(self):
    return self._X_val

  @property
  def y_val(self):
    return self._y_val

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def num_val_examples(self):
    return self._num_val_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def get_validation_set(self):
    return self._X_val, self._y_val
  
  def reset_val_index(self):
      self._index_in_val_epoch = 0
        
  
  def next_test_batch(self, batch_size, shuffle=True):
    start = self._index_in_test_set
    self._index_in_test_set += batch_size
    
    #we've reached the end of the test set
    if self._index_in_test_set > self._num_test_examples:
        end = self._num_test_examples
    else:
        end = self._index_in_test_set

    imgs = load_images_ooc.load_next_batch(self._test_files[start:end], width=self._width, height=self._height)
    file_names = [os.path.basename(fl) for fl in self._test_files[start:end]]
    return (imgs, file_names)


  def next_val_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_val_epoch
    # Shuffle for the first epoch
    if self._val_epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_val_examples)
      np.random.shuffle(perm0)
      self._X_val = self._X_val[perm0]
      self._y_val = self._y_val[perm0]
    self._index_in_val_epoch += batch_size
    end = self._index_in_val_epoch
    imgs = load_images_ooc.load_next_batch(self._X_val[start:end], width=self._width, height=self._height)
    labels_encoded = load_images_ooc.one_hot_encode(self._y_val[start:end])
    return (imgs, labels_encoded)
    
  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._X_train = self._X_train[perm0]
      self._y_train = self._y_train[perm0]
    
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      train_rest_part = self._X_train[start:self._num_examples]
      labels_rest_part = self._y_train[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._X_train = self._X_train[perm]
        self._y_train = self._y_train[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      train_new_part = self._X_train[start:end]
      labels_new_part = self._y_train[start:end]
      img_to_load = np.concatenate((train_rest_part, train_new_part), axis=0)
      labels_to_encode = np.concatenate((labels_rest_part, labels_new_part), axis=0)
      imgs = load_images_ooc.load_next_batch(img_to_load, width=self._width, height=self._height)
      labels_encoded = load_images_ooc.one_hot_encode(labels_to_encode)
      return imgs, labels_encoded
      
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      imgs = load_images_ooc.load_next_batch(self._X_train[start:end], width=self._width, height=self._height)
      labels_encoded = load_images_ooc.one_hot_encode(self._y_train[start:end])
      return imgs, labels_encoded