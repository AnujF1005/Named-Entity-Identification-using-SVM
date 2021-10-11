import pickle
import glob
import os

def get_gazetteer():

  gaz = []
  if os.path.isfile('gazetteer/gazetteer'):
    with open('gazetteer/gazetteer','rb') as f:
      gaz = pickle.load(f)
    return gaz
    
  path = "gazetteer/*.txt"
  print('Creating Gazetteer')
  for fl in glob.glob(path):
      with open(fl, 'r') as f:
        while True:
          try:
            line = f.readline()
            if not line:
              break
            line = line.replace('\n','')
            if len(line) > 0:
              gaz.append(line)
          except:
            continue
  gaz = list(set(gaz))
  
  #Remove \n
  #gaz = [i.replace('\n','') for i in gaz]
  
  with open('gazetteer/gazetteer','wb') as f:
    pickle.dump(gaz,f)
  
  return gaz