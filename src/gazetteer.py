import pickle
import glob
import os

def get_gazetteer():

  gaz = []
  if os.path.isfile('gazetteer\gazetteer'):
    with open('gazetteer\gazetteer','rb') as f:
      gaz = pickle.load(f)
    return gaz
    
  path = "gazetteer\*.txt"

  for file in glob.glob(path):
    with open(file, 'r') as f:
      gaz += f.readlines()
  gaz = list(set(gaz))
  
  #Remove \n
  gaz = [i.replace('\n','') for i in gaz]
  
  with open('gazetteer\gazetteer','wb') as f:
    pickle.dump(gaz,f)
  
  return gaz