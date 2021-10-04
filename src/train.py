from sklearn import svm
from dataset import get_train
from validate_modules import validate
import pickle

validate()

print('Loading data..')

train_X, train_y = get_train()

print('Data is successfully loaded.')

classifier = svm.LinearSVC()

print("Training started...")
classifier.fit(train_X, train_y)

with open('models/svm','wb') as f:
  pickle.dump(classifier, f)

print("Trainined ended! Trained model is saved at models/svm")