# Evaluate the model

# Import libraries
import sys
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve

from model import LSAembed
from utils import prepare_data
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer


def evaluate(model_name: str,
             n_feature: List[int],
             text_train, text_test, y_train, y_test,
             save: bool = False):
    
    
    train_accuracy, test_accuracy = [], []
    train_roc_auc, test_roc_auc = [], []
    
    train_svd_accuracy, test_svd_accuracy = [], []
    train_svd_roc_auc, test_svd_roc_auc = [], []
    
    # Initialize the model
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=500)
        
        for n in tqdm(n_feature):
            # Initialize the vectorizer
            vectorizer = TfidfVectorizer(tokenizer= None,
                                         preprocessor= None,
                                         max_features= n)
            
            # Vectorize the text data
            X_train = vectorizer.fit_transform(text_train)
            X_test = vectorizer.transform(text_test)
            
            # Fit the model    
            model.fit(X_train, y_train)
            
            # Accuracy
            train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))
            test_accuracy.append(accuracy_score(y_test, model.predict(X_test)))
            
            # ROC AUC
            Pr_train = model.predict_proba(X_train)
            Pr_test = model.predict_proba(X_test)
            train_roc_auc.append(roc_auc_score(y_train, Pr_train, multi_class='ovo'))
            test_roc_auc.append(roc_auc_score(y_test, Pr_test, multi_class='ovo'))  
            
        # Initialize the LSA model
        vectorizer = TfidfVectorizer(tokenizer= None,
                                     preprocessor= None,
                                     max_features= n_feature[-1])
        X_train = vectorizer.fit_transform(text_train)
        X_test = vectorizer.transform(text_test)
        
        for n in tqdm(n_feature):
            # Initialize the LSA model
            lsa = LSAembed(model_name=model_name, n_components=n)
            
            # Fit the model
            lsa.fit(X_train, y_train)
            
            # Accuracy
            train_svd_accuracy.append(lsa.score(X_train, y_train))
            test_svd_accuracy.append(lsa.score(X_test, y_test))
            
            # ROC AUC
            Pr_train = lsa.predict_proba(X_train)
            Pr_test = lsa.predict_proba(X_test)
            train_svd_roc_auc.append(roc_auc_score(y_train, Pr_train, multi_class='ovo'))
            test_svd_roc_auc.append(roc_auc_score(y_test, Pr_test, multi_class='ovo'))      
    
    
    # Plot the results
    cmap = cm.get_cmap('viridis', 2)
    markers = ['o', 's']    
    fontsize = 15
    
    # Test Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Test Accuracy', fontsize= fontsize)
    plt.xlabel('Number of Features', fontsize= fontsize)
    plt.ylabel('Accuracy', fontsize= fontsize)
    plt.plot(n_feature, test_accuracy, label=model_name, 
             color=cmap(0), marker= markers[0])
    plt.plot(n_feature, test_svd_accuracy, label='LSA+' + model_name,
             color=cmap(1), marker= markers[1])
    plt.grid()
    plt.legend()
    
    # Test ROC AUC
    plt.subplot(1, 2, 2)
    plt.title('Test ROC AUC', fontsize= fontsize)
    plt.xlabel('Number of Features', fontsize= fontsize)
    plt.ylabel('ROC AUC', fontsize= fontsize)
    plt.plot(n_feature, test_roc_auc, label=model_name,
             color=cmap(0), marker= markers[0])
    plt.plot(n_feature, test_svd_roc_auc, label='LSA+' + model_name,
             color=cmap(1), marker= markers[1])
    plt.grid()
    plt.legend()
    
    # Save the figure
    if save:
        # Add the path to the src directory
        sys.path.append(os.path.abspath(os.path.join('..', 'plots')))
        plt.tight_layout()
        plt.savefig(os.path.join('..', 'plots', model_name + '_test.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # Train Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Train Accuracy', fontsize= fontsize)
    plt.xlabel('Number of Features', fontsize= fontsize)
    plt.ylabel('Accuracy', fontsize= fontsize)
    plt.plot(n_feature, train_accuracy, label= model_name,
             color=cmap(0), marker= markers[0])
    plt.plot(n_feature, train_svd_accuracy, label='LSA+' + model_name,
             color=cmap(1), marker= markers[1])
    plt.grid()
    plt.legend()
    
    # Train ROC AUC
    plt.subplot(1, 2, 2)
    plt.title('Train ROC AUC', fontsize= fontsize)
    plt.xlabel('Number of Features', fontsize= fontsize)
    plt.ylabel('ROC AUC', fontsize= fontsize)
    plt.plot(n_feature, train_roc_auc, label=model_name,
             color=cmap(0), marker= markers[0])
    plt.plot(n_feature, train_svd_roc_auc, label='LSA+' + model_name,
             color=cmap(1), marker= markers[1])
    plt.grid()
    plt.legend()
    
    # Save the figure
    if save:
        # Add the path to the src directory
        sys.path.append(os.path.abspath(os.path.join('..', 'plots')))
        plt.tight_layout()
        plt.savefig(os.path.join('..', 'plots', model_name + '_train.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    