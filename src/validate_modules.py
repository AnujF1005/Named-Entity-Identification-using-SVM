import nltk

def validate():
  """
  Check whether all required modules are available or not.
  """
  nltk.download('wordnet')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('tagsets')
  nltk.download('stopwords')