# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:44:48 2020

@author: nishant
"""


import config
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    '''
    Function takes text as input and then remove comma and (fluently),(poorly),(okay) and 
    'language' from the text

    '''
    
    text = re.sub(',',' ',text)
    text = re.sub('language','',text)
    text = re.sub('\(([^)]+)\)','',text)
    
    text=re.sub(' +'," ",text)
    
    return text
  


def bag_of_words(data,col):
    '''
    This function helps to transform langauge using bag of words with each language as a feature
    
    '''
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[col]).toarray()
    columns=vectorizer.get_feature_names()   
    language_feature = pd.DataFrame(X,columns=columns)
    return language_feature

def tf_idf(data,col):
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    X = vectorizer.fit_transform(data[col].values.astype('U')).toarray()
    columns=vectorizer.get_feature_names()
    tf_idf_feat= pd.DataFrame(X,columns=columns)
    return tf_idf_feat




if __name__ == "__main__":
    
      df=pd.read_csv(config.TRAINING_FILE)
      df=df.drop(columns=['Unnamed: 0'])
      
      df['language']=df['language'].apply(lambda x:clean_text(x))
      
      lan_feature=tf_idf(df, "language")
      new_df=pd.concat([df,lan_feature],axis=1)
      new_df=new_df.drop(columns=['language'])
      
      new_df.to_csv('finalpreprocess_2')