import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.data import load
import numpy as np
# from gazetteer import get_gazetteer
import re
from sklearn.preprocessing import MinMaxScaler

def get_word_features(sentence, i, word_window_size=5, prevLabel=None):
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))

  features = {} #Dictionary to store features of each word

  curword = sentence['tokens'][i]

  features['isFirstWord'] = int(i==0)
  features['isLastWord'] = int(i==(len(sentence['tokens'])-1))
  features['allLower'] = int(curword.islower())
  features['allCaps'] = int(curword.isupper())
  features['firstCap'] = int(curword[0].isupper())
  features['isNum'] = int(curword.isdigit())
  features['isNoun'] = int(pos_tag([curword.lower()])[0][1].lower() in ['n','nn', 'nnp', 'nnps'])
  features['length'] = len(curword)
  features['wordposition'] = i
  features['twoDigitNum'] = int(curword.isdigit() and len(curword)==2)
  features['fourDigitNum'] = int(curword.isdigit() and len(curword)==4)
  features['containsDigit'] = int(any(char.isdigit() for char in curword))
  features['containsDigitAndAlpha'] = int(features['containsDigit'] and any(char.isalpha() for char in curword))
  features['containsDigitAndDash'] = int(features['containsDigit'] and ('-' in curword))
  features['containsDigitAndSlash'] = int(features['containsDigit'] and ('/' in curword))
  features['containsDigitAndComma'] = int(features['containsDigit'] and (',' in curword))
  features['containsDigitAndPeriod'] = int(features['containsDigit'] and ('.' in curword))
  features['isInitial'] = int(features['firstCap'] and len(curword)==2 and curword[1]=='.')
  features['isOther'] = int(all(not char.isalnum() for char in curword))
  if i > 0 and i < len(sentence['tokens'])-1:
    features['isStopWordAndSurroundByNoun'] = int((lemmatizer.lemmatize(curword) in stop_words) and (pos_tag([sentence['tokens'][i-1].lower()])[0][1].lower() in ['n','nn', 'nnp', 'nnps']) and (pos_tag([sentence['tokens'][i+1].lower()])[0][1].lower() in ['n','nn', 'nnp', 'nnps']))
  else:
    features['isStopWordAndSurroundByNoun']=0
  if prevLabel != None:
    features['isPrevNE'] = prevLabel
  elif i > 0:
    features['isPrevNE'] = int(any(char.isdigit() for char in sentence['tokens'][i-1]) or sentence['ner_tags'][i-1] != 0)
  else:
    features['isPrevNE'] = 0
  
  return features

def get_features(sentence, word_window_size=5, prevLabel=None):
  """
  Extract features from sentence.
  Parameters
  ----------
  sentence: (dict) Sentence in from of dict from Conll dataset
  Returns
  ----------
  1D list of feature dictionaries for each word
  """

  sentence_features = [] 
  
  for i in range(len(sentence['tokens'])):
    features = get_word_features(sentence, i)
    sentence_features.append(features)

  return sentence_features
  
def get_targets(sentence):
  """
  Extract targets (0: No name / 1: Name) from sentence.
  Parameters
  ----------
  sentence: (dict) Sentence in from of dict from Conll dataset
  Returns
  ----------
  1D list of targets(0/1) for each word
  """

  targets = []
  for i in range(len(sentence['tokens'])):
    if any(char.isdigit() for char in sentence['tokens'][i]):
      targets.append(1)
    elif sentence['ner_tags'][i] != 0:
      targets.append(1)
    else:
      targets.append(0)

  return targets
  
def preprocess(dataset, word_window_size=5, train_data=False, label=True,scaling=False):
  """
  Convert the textual features to numeric features.
  Parameters
  ----------
  dataset: (dataframe) Dataframe of dataset with column names as feature names
  word_window_size: (int) Neighbours for each word to consider from
  train_data: (bool) is given dataset is training data or not 
  Returns
  ----------
  1D list of feature dictionaries for each word
  """
  
  word2idx = None
  lemma2idx = None
  suffix22idx = None
  suffix32idx = None
  prefix22idx = None
  prefix32idx = None
  all_pos = list(load('help/tagsets/upenn_tagset.pickle').keys())
  pos2idx = {p:i+1 for i,p in enumerate(all_pos)}
  pos2idx['0'] = 0
  
  '''
  if train_data:
    words = []
    lemmas = []
    suffix2 = []
    suffix3 = []
    prefix2 = []
    prefix3 = []
    for i in range(-(word_window_size//2), word_window_size//2 + 1):
      words += list(dataset['Word_'+str(i)])
      lemmas += list(dataset['WordLemma_'+str(i)])
      suffix2 += list(dataset['Suffix2_'+str(i)])
      suffix3 += list(dataset['Suffix3_'+str(i)])
      prefix2 += list(dataset['Prefix2_'+str(i)])
      prefix3 += list(dataset['Prefix3_'+str(i)])
    words = list(set(words))
    lemmas = list(set(lemmas))
    suffix2 = list(set(suffix2))
    suffix3 = list(set(suffix3))
    prefix2 = list(set(prefix2))
    prefix3 = list(set(prefix3))
    word2idx = {w:i+1  for i,w in enumerate(words)}
    lemma2idx = {w:i+1  for i,w in enumerate(lemmas)}
    suffix22idx = {w:i+1  for i,w in enumerate(suffix2)}
    suffix32idx = {w:i+1  for i,w in enumerate(suffix3)}
    prefix22idx = {w:i+1  for i,w in enumerate(prefix2)}
    prefix32idx = {w:i+1  for i,w in enumerate(prefix3)}
    
    with open('preprocessing_assets/data_dictionaries', 'wb') as f:
      pickle.dump([word2idx, lemma2idx, suffix22idx, suffix32idx, prefix22idx, prefix32idx], f)
    
  else:
    with open('preprocessing_assets/data_dictionaries', 'rb') as f:
      word2idx, lemma2idx, suffix22idx, suffix32idx, prefix22idx, prefix32idx=pickle.load(f)
  '''
  X,y = [], []

  for rec in range(dataset.shape[0]):
  
    if label:
      y.append(dataset.loc[rec, 'y'])

    x=[]
    for col in dataset.columns.tolist():
      if col == 'y':
        continue

      if col.startswith('POS_'):
        x.append(pos2idx[dataset.loc[rec,col]])
      elif isinstance(dataset.loc[rec,col],str):
        try:
          x += list(model_w2v[dataset.loc[rec,col]])[:10]
        except:
          x += [0]*10
      else:
        x.append(dataset.loc[rec,col])
      '''
      if col.startswith('Word_'):
        if (dataset.loc[rec, col] not in word2idx):
          x.append(0)
        else:
          x.append(word2idx[dataset.loc[rec, col]])
      elif col.startswith('WordLemma_'):
        if (dataset.loc[rec, col] not in lemma2idx):
          x.append(0)
        else:
          x.append(lemma2idx[dataset.loc[rec,col]])
      elif col.startswith('Suffix2_'):
        if (dataset.loc[rec, col] not in suffix22idx):
          x.append(0)
        else:
          x.append(suffix22idx[dataset.loc[rec,col]])
      elif col.startswith('Suffix3_'):
        if (dataset.loc[rec, col] not in suffix32idx):
          x.append(0)
        else:
          x.append(suffix32idx[dataset.loc[rec,col]])
      elif col.startswith('Prefix2_'):
        if (dataset.loc[rec, col] not in prefix22idx):
          x.append(0)
        else:
          x.append(prefix22idx[dataset.loc[rec,col]])
      elif col.startswith('Prefix3_'):
        if (dataset.loc[rec, col] not in prefix32idx):
          x.append(0)
        else:
          x.append(prefix32idx[dataset.loc[rec,col]])
      elif col.startswith('POS_'):
        x.append(pos2idx[dataset.loc[rec,col]])
      else:
        x.append(dataset.loc[rec,col])
      '''
    X.append(x)

  
  X,y = np.array(X), np.array(y)
  
  #print(X.sum(axis=0))
  #print(X[y[:]==1].sum(axis=0))
  
  
  if train_data:
    #Handling Imbalance  
    p = np.random.permutation(X.shape[0])
    #Shuffling
    X,y = X[p], y[p]

    mask = y[:]==1
    new_X = X[mask]
    new_y = y[mask]
    count = 2*new_X.shape[0]
    mask = np.logical_not(mask)
    new_X = np.vstack([new_X, X[mask][:count]])
    new_y = np.array(list(new_y)+ list(y[mask][:count]))
    X = new_X
    y = new_y

    p = np.random.permutation(X.shape[0])
    #Shuffling
    X,y = X[p], y[p]
    ####
  
  #print(X.sum(axis=0))
  #print(X[y[:]==1].sum(axis=0))

  if scaling:
    if train_data:
      scaler = MinMaxScaler()
      X = scaler.fit_transform(X)
      with open('preprocessing_assets/scaler', 'wb') as f:
        pickle.dump(scaler, f)
    else:
      with open('preprocessing_assets/scaler', 'rb') as f:
        scaler = pickle.load(f)
        X = scaler.transform(X)
      
  
  return X,y
  
def summary(X,y):
  print("-----Summary of dataset-----")
  print("Total number of samples:", X.shape[0])
  print("Total number of samples with label 0:", y[y[:]==0].shape[0])
  print("Total number of samples with label 1:", y[y[:]==1].shape[0])
  print("----------------------------")
  print()