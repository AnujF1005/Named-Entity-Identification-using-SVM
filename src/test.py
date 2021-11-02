from sklearn import svm
import os
import pickle
from preprocess import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def refine(sentence, tokens):
  stop_words = set(stopwords.words('english'))
  lemmatizer = WordNetLemmatizer()

  for i in range(1,len(sentence)-1):
    if tokens[i-1]==1 and tokens[i+1]==1:
      if lemmatizer.lemmatize(sentence[i].lower()) in stop_words:
        tokens[i]=1
  return tokens

model_path = 'models/svm'

if not os.path.isfile(model_path):
  print("No trained model found at models/.. Please train model first using train.py!");
  exit()

classifier = None
with open(model_path,'rb') as f:
  classifier = pickle.load(f)

while True:
  s = input('Enter the sentence: ').strip()
  #s = s.split(' ')
  s = word_tokenize(s)

  query = {'tokens': s}
  prevLabel=None
  tokens = []
  
  for i in range(len(s)):
    q,_ = preprocess(pd.DataFrame([get_word_features(query, i, prevLabel=prevLabel)]), train_data=False, label=False, scaling=True)
    prevLabel = classifier.predict(q)[0]
    tokens.append(prevLabel)
    
  tokens = refine(s, tokens)

  result = ""
  for w,t in zip(s, tokens):
    result += w + "_" + str(int(t)) + " "
  print(result)
  print()