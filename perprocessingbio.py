# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:18:46 2020

@author: nishant
"""
#NLP Featurization
import os
import pandas as pd
import numpy as np
import config
import spacy 
import nltk
from tqdm import tqdm
import re
import string
from spacy.lang.en import English

def clean_text(text):
    '''
    looking for punctuations and html tags and cleaning them from the 
    text
    
    '''
    
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text

contractions = {"ain't": "am not / are not","aren't": "are not / am not","can't": "cannot","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not",
"doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not",
"he'd": "he had / he would","he'd've": "he would have","he'll": "he shall / he will","he'll've": "he shall have / he will have",
"he's": "he has / he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how has / how is","i'd": "I had / I would",
"i'd've": "I would have","i'll": "I shall / I will","i'll've": "I shall have / I will have",
"i'm": "I am","i've": "I have","isn't": "is not","it'd": "it had / it would","it'd've": "it would have",
"it'll": "it shall / it will","it'll've": "it shall have / it will have","it's": "it has / it is","let's": "let us",
"ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have",
"mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have",
"o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","she'd": "she had / she would","she'd've": "she would have","she'll": "she shall / she will","she'll've": "she shall have / she will have",
"she's": "she has / she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have",
"so's": "so as / so is","that'd": "that would / that had","that'd've": "that would have","that's": "that has / that is","there'd": "there had / there would",
"there'd've": "there would have","there's": "there has / there is","they'd": "they had / they would","they'd've": "they would have",
"they'll": "they shall / they will","they'll've": "they shall have / they will have","they're": "they are","they've": "they have",
"to've": "to have","wasn't": "was not","we'd": "we had / we would","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
"we're": "we are","we've": "we have","weren't": "were not","what'll": "what shall / what will","what'll've": "what shall have / what will have",
"what're": "what are","what's": "what has / what is","what've": "what have","when's": "when has / when is",
"when've": "when have","where'd": "where did","where's": "where has / where is",
"where've": "where have","who'll": "who shall / who will","who'll've": "who shall have / who will have","who's": "who has / who is","who've": "who have",
"why's": "why has / why is","why've": "why have","will've": "will have","won't": "will not",
"won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you had / you would","you'd've": "you would have","you'll": "you shall / you will","you'll've": "you shall have / you will have",
"you're": "you are","you've": "you have"
}

def remove_contraction(sentence):
    '''
    all the contractions in the sentence are removed 
    
    '''
    for word in sentence.split():
        if word.lower() in contractions:
           sentence = sentence.replace(word, contractions[word.lower()]) 
    return sentence  


    
    


def text_preprocess(text):
    '''
    An interface to remove all the contractions in the sentence and then 
    cleaning the text from any type of typo error
    
    '''
    text=remove_contraction(text)
    text=clean_text(text)
    return text


def remove_stopwords(token):
    '''
    Function takes text input and removes the stopwords and lemmatizes the words
    
    '''
    doc = spacy_nlp(token)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    text = str()
    
    for word in tokens:
        text += word
        if(text[-1]!=' '):
            text += " "
        
    text=re.sub(' +', ' ', text)  
    return text



if __name__=='__main__':
    df=pd.read_csv(config.TRAINING_FILE)
    df=df.drop(columns=['Unnamed: 0'])
    #Cleaning data
    df['bio']=df['bio'].apply(lambda x:text_preprocess(x))
    
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    #Removing Stopwords
    df['bio']=df['bio'].apply(lambda x:remove_stopwords(x))
    
    df.to_csv('preprocessbio_2')
    
    