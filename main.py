# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:47:40 2020

@author: -
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import re

# Find all available data
def find_all_spreadsheets():
    available_data = {}
    r = requests.get('https://www.football-data.co.uk/downloadm.php')
    if r.status_code != 200:
        print('Oh dear. Error {}'.format(r.status_code))
        return -1
    matches = re.findall('(mmz(.*?)[0-9]+-[0-9]+(.*?).[xls]+")',r.text)
    for match in matches:
        tmp = match[0].replace('"','')
        season = re.search('[0-9]+-[0-9]+',tmp).group()
        available_data[season] = tmp
    return available_data


def load_data(data_url):
    data = pd.read_csv(data_url)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data = data.drop(['date','time'],axis=1)
    # Rearrange columns
    cols = data.columns.tolist()
    cols.remove('datetime')
    cols.insert(0,'datetime')
    data = data[cols]
    return data


def season_spreadsheet(data_url):
    url = 'https://www.football-data.co.uk/'+data_url
    data = pd.read_excel(url,sheet_name=None)
    return data

spreadsheets = find_all_spreadsheets()

data_state = st.text('Nothing is happening')

@st.cache
def load_all_data(spreadsheets):
    # Silly. But it works..
    base_url = 'https://www.football-data.co.uk/'
    big_df = pd.DataFrame()
    all_keys = list(spreadsheets.keys())
    stop_season = '2009-2010'
    stop_season = '2015-2016'
    pos = all_keys.index(stop_season)
    for c,key in enumerate(all_keys):
        
        if key == stop_season:
            break
        
        data_state.text('Loading season {} ... only {} left to go'.format(key,pos-c))
        url = base_url + spreadsheets[key]
        og_spreadsheet = pd.read_excel(url,None)
        
        
        big_spreadsheet = pd.concat(og_spreadsheet, ignore_index=True)
        big_spreadsheet['season'] = key
        big_df = big_df.append(big_spreadsheet, sort=False,ignore_index=True)
    return big_df

big_df = load_all_data(spreadsheets)
data_state.text('')

#full_spreadsheet = season_spreadsheet(spreadsheets['2019-2020'])
#full_df = full_spreadsheet['E0']
#
season = st.multiselect(
    "Select season", list(big_df['season'].unique()),['2019-2020']
)

division = st.multiselect(
    "Select division", list(big_df['Div'].unique()),['E0']
)


data = big_df.loc[big_df['Div'].isin( division) & big_df['season'].isin(season)]

data=data.dropna(axis=1,how='all')

columns = st.multiselect(
    "Select columns", list(data.columns),['Div','HomeTeam']
)

data = data[columns]

st.write("### Raw data", data.sort_index())

## Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading data...')
## Load 10,000 rows of data into the dataframe.
#data = load_data(5)
## Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading data...done!')
#
#st.subheader('Raw data')
#st.write(data)


#def load_data(season='2019-2020')#



