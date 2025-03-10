from sklearn import svm
from dataset import get_test
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model_path = 'models/svm'

if not os.path.isfile(model_path):
  print("No trained model found at models/.. Please train model first using train.py!");
  exit()


test_X, test_y = get_test()

classifier = None
with open(model_path,'rb') as f:
  classifier = pickle.load(f)

pred_y = classifier.predict(test_X)
print(classification_report(test_y, pred_y))

cf_matrix = confusion_matrix(test_y, pred_y)
sns.heatmap(cf_matrix, annot=True)
plt.savefig('confusion_matrix.png')