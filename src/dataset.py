from datasets import load_dataset
import pandas as pd
import pickle
from preprocess import *
import os

def get_train():
  
  file_name = 'dataset/train_dataset'
  X, y = [], []
  
  if os.path.isfile(file_name):
    with open(file_name,'rb') as f:
      X, y = pickle.load(f)
    print("Training data:")
    print(summary(X,y))
    return X,y
  
  conll_data = load_dataset('conll2003')
  
  print('Extracting Features...')
  for rec in conll_data['train']:
    X+=get_features(rec)
    y+=get_targets(rec)

  dataset = pd.DataFrame(X)
  dataset['y'] = y
  
  print('Preprocessing Features...')
  X,y= preprocess(dataset, train_data=True, scaling=True)
  with open(file_name,'wb') as f:
    pickle.dump([X,y], f)
    
  print("Training data:")
  print(summary(X,y))
  return X,y
  
def get_test():
  file_name = 'dataset/test_dataset'
  X, y = [], []
  
  if os.path.isfile(file_name):
    with open(file_name,'rb') as f:
      X, y = pickle.load(f)
      
    print("Testing data:")
    print(summary(X,y))
    return X,y
  
  conll_data = load_dataset('conll2003')
  
  print('Extracting Features...')
  for rec in conll_data['test']:
    X+=get_features(rec)
    y+=get_targets(rec)

  dataset = pd.DataFrame(X)
  dataset['y'] = y
  
  print('Preprocessing Features...')
  X,y= preprocess(dataset, scaling=True)
  with open(file_name,'wb') as f:
    pickle.dump([X,y], f)
  
  print("Testing data:")
  print(summary(X,y))
  
  return X,y