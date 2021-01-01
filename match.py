# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:59:35 2020

@author: nishant
"""
#match
import pandas as pd
import numpy as np
import config


if __name__ == "__main__":
    df=pd.read_csv(config.TRAINING_FILE)
    df=df.drop(columns=['Unnamed: 0'])
    
    
    df=df.set_index('user_id')
    corr_group = df.T.corr()
    corr_group[corr_group < 0] = 0
    
    list_to_zero=pd.read_csv(config.LIST_FILE)
    
    for i in range(len(list_to_zero)):
        corr_group[list_to_zero['user_id_1'][i]][list_to_zero['user_id_2'][i]]=0
        corr_group[list_to_zero['user_id_2'][i]][list_to_zero['user_id_1'][i]]=0

    for i in corr_group.columns:
        corr_group[i][i]=0
        
    
    
    
    
    corr_group.to_csv('submission.csv',index=True)
    