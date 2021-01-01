# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:42:59 2020

@author: nishant
"""
#feature_extract.py

import config
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import models
from tqdm import tqdm

def tf_idf(data,col):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(df['bio'].values.astype('U')).toarray()
    columns=vectorizer.get_feature_names()
    tf_idf_feat= pd.DataFrame(X,columns=columns)
    return tf_idf_feat
 
def bag_of_words(data,col):
    '''
    This function helps to transform langauge using bag of words with each language as a feature
    
    '''
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[col].values.astype('U')).toarray()
    columns=vectorizer.get_feature_names()   
    language_feature = pd.DataFrame(X,columns=columns)
    return language_feature
   

def tf_idf_w2v(data,col):
    '''
    Convert every data in the corpus as an vector using tf-idf and word2vec
    '''
    w2v_model = models.KeyedVectors.load_word2vec_format('C:/Users/Nishant/HackerEarth Competition/src/GoogleNews-vectors-negative300.bin.gz', binary=True)
    w2v_words = list(w2v_model.wv.vocab)
    model = TfidfVectorizer()
    model.fit(data[col].values.astype('U'))
    dictionary = dict(zip(model.get_feature_names(),list(model.idf_)))
    
    tf_idf_feat = model.get_feature_names()
    tfidf_sent_vectors = []
    row = 0
    for sent in tqdm(data[col]):
    
        sent_vec = np.zeros(50)
        weight_sum = 0
        
        for word in str(sent) :
            if word in w2v_words and word in tf_idf_feat :
                    vec = w2v_model.wv[word]
                    tf_idf = dictionary[word]*(sent.count(word)/len(sent))
                    sent_vec += ( vec*tf_idf )
                    weight_sum += tf_idf
        if weight_sum != 0:
            sent_vec /= weight_sum
                    
        tfidf_sent_vectors.append(sent_vec)
        row += 1
        
    new_df_feat = pd.DataFrame(tfidf_sent_vectors)
    return new_df_feat
    
    

if __name__ == "__main__":
    
    df=pd.read_csv(config.TRAINING_FILE)
    df=df.drop(columns=['Unnamed: 0'])
    
    tf_idf_feat = tf_idf_w2v(df,"bio")
    
    new_df=pd.concat([df,tf_idf_feat],axis=1)
    new_df=new_df.drop(columns=['bio'])
    
    #bag_feat= bag_of_words(df,"bio")
    #new_df=pd.concat([df,bag_feat],axis=1)
    #new_df=new_df.drop(columns=['bio'])
    
    new_df.to_csv('finaldata_tfidf_w2v')
    
    

