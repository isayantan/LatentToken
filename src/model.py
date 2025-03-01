# Description: Train model

# Import libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD


class LSAembed:
    def __init__(self,
                 model_name,
                 n_components=100):
        
        if model_name == 'MultinomialNB':
            self.model = MultinomialNB() 
        elif model_name == 'LogisticRegression':
            self.model = LogisticRegression(max_iter = 500)
        elif model_name == 'RandomForestClassifier':
            self.model = RandomForestClassifier()
        
        self.n_components = n_components      
        self.svd = TruncatedSVD(n_components= self.n_components)        

    def fit(self, X, y):
        # Fit the model
        Z = self.svd.fit_transform(X)
        self.model.fit(Z, y)

    def predict(self, X):
        # Predict
        Z = self.svd.transform(X)
        pred = self.model.predict(Z)
        return pred
    
    def score(self, X, y):
        # Score
        Z = self.svd.transform(X)
        score = self.model.score(Z, y)
        return score
    
    def predict_proba(self, X):
        # Predict probabilities
        Z = self.svd.transform(X)
        proba = self.model.predict_proba(Z)
        return proba