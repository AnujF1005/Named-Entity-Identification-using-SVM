from sklearn import svm
import os
import pickle
from preprocess import *

model_path = 'models/svm'

if not os.path.isfile(model_path):
  print("No trained model found at models/.. Please train model first using train.py!");
  exit()

classifier = None
with open(model_path,'rb') as f:
  classifier = pickle.load(f)

while True:
  s = input('Enter the sentence: ').strip()
  s = s.split(' ')
  
  query = {'tokens': s}
  query,_ = preprocess(pd.DataFrame(get_features(query)), train_data=False, label=False, scaling=True)
  
  pred = classifier.predict(query)
  
  result = ''
  for word, tag in zip(s, pred):
    result += word + '_' + str(tag) + ' '
  
  print("Prediction:", result)
  print()