# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:47:40 2020

@author: -
"""

import streamlit as st
import numpy as np
import pandas as pd

st.title('Dashboard')

DATE_COLUMN = 'Date'
DATA_URL = ('https://www.football-data.co.uk/mmz4281/2021/E3.csv'
         #'streamlit-demo-data/uber-raw-data-sep14.csv.gz'
         )

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data = data.drop(['date','time'],axis=1)
    return data

d=load_data(5)
d