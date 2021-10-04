import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.data import load
import numpy as np
from gazetteer import get_gazetteer
import re
from sklearn.preprocessing import MinMaxScaler

class Scaler():
  def __call__(self,features, is_train=False):
    if is_train:
      self.mins= features.min(axis=0)
      self.maxes = features.max(axis=0)

    scaled_features = (features-self.mins)/(self.maxes-self.mins)
        
    return scaled_features

def get_features(sentence, word_window_size=5):
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
  
  lemmatizer = WordNetLemmatizer()
  #gaz = get_gazetteer()

  for i in range(len(sentence['tokens'])):
    features = {} #Dictionary to store features of each word

    #Chunk tag of word
    #features['Chunk'] = sentence['chunk_tags'][i]
    
    ''' 
    #Is it in gazetteer
    score = 0
    score += int(sentence['tokens'][i].lower() in gaz)
    if i-1 >= 0 :
      score += int((' '.join(sentence['tokens'][i-1:i+1])).lower() in gaz)
    if i+1 < len(sentence['tokens']):
      score += int((' '.join(sentence['tokens'][i:i+2])).lower() in gaz)
    
    features['isInGazetteer'] = int(score>0)
    '''

    #Whether first character is capital or not
    features['firstCap'] = int(sentence['tokens'][i][0].isupper())

    #Whether it is first word or not
    features['firstWord'] = int(i==0)

    #Whether all characters of word are capital or not
    features['allCaps'] = int(sentence['tokens'][i].isupper())
    
    features['containsDigitandAlpha'] = int(bool(re.search(r'\d', sentence['tokens'][i])))
    
    features['isDigit'] = int(sentence['tokens'][i].isdigit())

    #features['isSingleCharWord'] = int(len(sentence['tokens'][i]) == 1)

    #features['isInitial'] = int(sentence['tokens'][i][-1] == '.' and len(sentence['tokens'][i])==2 and sentence['tokens'][i][0].isalpha())

    itr = i - word_window_size // 2
    
    while itr <= i+word_window_size//2:
      
      #Word
      if itr<0 or itr >= len(sentence['tokens']):
        features['Word_'+str(itr-i)] = '<PAD>'
      else:
        features['Word_'+str(itr-i)] = sentence['tokens'][itr].lower()

      #Word lemma
      if itr<0 or itr >= len(sentence['tokens']):
        features['WordLemma_'+str(itr-i)] = '<PAD>'
      else:
        features['WordLemma_'+str(itr-i)] = lemmatizer.lemmatize(features['Word_'+str(itr-i)])

      #POS tag
      if itr<0 or itr >= len(sentence['tokens']):
        features['POS_'+str(itr-i)] = '<PAD>'
      else:
        #features['POS_'+str(itr-i)] = sentence['pos_tags'][itr]
        features['POS_'+str(itr-i)] = pos_tag([sentence['tokens'][itr]])[0][1]

      #Suffix of length 2
      if itr<0 or itr >= len(sentence['tokens']):
        features['Suffix2_'+str(itr-i)] = '<PAD>'
      elif len(sentence['tokens'][itr]) >= 2:
        features['Suffix2_'+str(itr-i)] = sentence['tokens'][itr][-2:].lower()
      else:
        features['Suffix2_'+str(itr-i)] = '<PAD>'
      
      #Suffix of length 3
      if itr<0 or itr >= len(sentence['tokens']):
        features['Suffix3_'+str(itr-i)] = '<PAD>'
      elif len(sentence['tokens'][itr]) >= 3:
        features['Suffix3_'+str(itr-i)] = sentence['tokens'][itr][-3:].lower()
      else:
        features['Suffix3_'+str(itr-i)] = '<PAD>'
      
      #Prefix of length 2
      if itr<0 or itr >= len(sentence['tokens']):
        features['Prefix2_'+str(itr-i)] = '<PAD>'
      elif len(sentence['tokens'][itr]) >= 2:
        features['Prefix2_'+str(itr-i)] = sentence['tokens'][itr][:2].lower()
      else:
        features['Prefix2_'+str(itr-i)] = '<PAD>'
      
      #Prefix of length 3
      if itr<0 or itr >= len(sentence['tokens']):
        features['Prefix3_'+str(itr-i)] = '<PAD>'
      elif len(sentence['tokens'][itr]) >= 3:
        features['Prefix3_'+str(itr-i)] = sentence['tokens'][itr][:3].lower()
      else:
        features['Prefix3_'+str(itr-i)] = '<PAD>'
      
      
      itr+=1
      
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
    if sentence['ner_tags'][i] != 0:
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
  pos2idx['<PAD>'] = 0
  
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

  X,y = [], []

  for rec in range(dataset.shape[0]):
  
    if label:
      y.append(dataset.loc[rec, 'y'])

    x=[]
    for col in dataset.columns.tolist():
      if col == 'y':
        continue
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
    
    X.append(x)
  
  if scaling:
    if train_data:
      scaler = MinMaxScaler()
      X = scaler.fit_transform(np.array(X))
      with open('preprocessing_assets/scaler', 'wb') as f:
        pickle.dump(scaler, f)
    else:
      with open('preprocessing_assets/scaler', 'rb') as f:
        scaler = pickle.load(f)
        X = scaler.transform(np.array(X))
      
  
  return X,np.array(y)