import torch

from .svm import SVM
from .TLiDB_model import TLiDB_model
from tlidb.examples.utils import concat_t_d
from sklearn.feature_extraction.text import CountVectorizer

class BagOfWords(SVM):
    def __init__(self, config):
        self.config = config
        self.vectorizer = CountVectorizer()
        
