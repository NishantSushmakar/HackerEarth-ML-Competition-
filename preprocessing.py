# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:58:39 2020

@author: nishant
"""
import os 
import config
import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.preprocessing import OneHotEncoder


def pets(data,col):
    '''
    Function is used to map the dictionary values to the pets columns and if the person likes a cat or dog 
    then it is given a 1 , when it is not stated in the anything about  animal then 0, and if he/she dislikes them then 
    it is given -1 
    
    '''
    dict_possible = {'likes dogs and likes cats' : [1,1],'likes dogs':[1,0],'likes dogs and has cats':[1,1],
                       "has dogs"  :[1,0], 'has dogs and likes cats':[1,1], 'likes dogs and dislikes cats':[1,-1], 
            "has dogs and has cats":[1,1],"has cats":[0,1],"likes cats":[0,1],"has dogs and dislikes cats":[1,-1],
             "dislikes dogs and dislikes cats":[-1,-1],"dislikes dogs and likes cats":[-1,1],"dislikes cats":[0,-1],
             "dislikes dogs":[-1,0],"dislikes dogs and has cats":[-1,0]            
        }
    feature_pets =[]
    for text in df[col]:
        feature_pets.append(dict_possible[text])
        
        
    return feature_pets


def featurise_cat(data,cat_col):
    
    '''
    This function helps to featurise categoical columns using one hot encoding and then return a dataframe to be 
    inserted in to the main dataframe
    
    
    '''    
    ohe=OneHotEncoder()
    ans=ohe.fit_transform(data[cat_col]).toarray()
    columns=ohe.get_feature_names(cat_col)
    feature_cat=pd.DataFrame(ans,columns=columns)
    return feature_cat
        
        

def preprocess(data):
    '''
    Preprocess the categorical columns and columns other than the text data
    
    
    ''' 
    #Featurise Pets Column
    t = pets(data,"pets")
    feature_pets = pd.DataFrame(t, columns =['Dogs', 'Cats'])
    
    #Featurise categorical Columns by One Hot Encoding
    cat_col=['status','sex','orientation','drinks','drugs','job','smokes','new_languages',
             'body_profile','education_level','dropped_out','location_preference','interests','other_interests']    
    
    feature_cat = featurise_cat(data,cat_col)
    
    
    
    
    
    
    
    new_df=pd.concat([df,feature_pets,feature_cat],axis=1)
    new_df=new_df.drop(columns=['pets','status','sex','orientation','drinks','drugs','job','smokes','new_languages',
             'body_profile','education_level','dropped_out','location_preference','interests','other_interests','username','location'])
    
    
    return new_df


if __name__ == "__main__":
    df=pd.read_csv(config.TRAINING_FILE)
    df=df.drop(columns=['Unnamed: 0'])
    
    new_df=preprocess(df)
    
    new_df.to_csv('preprocessdata')
    
    