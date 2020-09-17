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
import altair as alt

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

data_state = st.text('')

@st.cache
def load_all_data(spreadsheets):
    # Silly. But it works..
    base_url = 'https://www.football-data.co.uk/'
    big_df = pd.DataFrame()
    all_keys = list(spreadsheets.keys())
    stop_season = '2009-2010'
#    stop_season = '2015-2016'
    pos = all_keys.index(stop_season)
    for c,key in enumerate(all_keys):
        print(key)
        
        if key == stop_season:
            break
        
        data_state.text('Loading season {} ... only {} left to go'.format(key,pos-c))
        url = base_url + spreadsheets[key]
        og_spreadsheet = pd.read_excel(url,None)
        
        
        big_spreadsheet = pd.concat(og_spreadsheet, ignore_index=True)
        big_spreadsheet['season'] = key
        big_spreadsheet['s-year'] = key[5:]
        big_spreadsheet['s-year'] = big_spreadsheet['s-year'].astype(int)
        if 'AG' in big_df.columns:
            big_spreadsheet['total-goals'] = big_spreadsheet['HG'] + big_spreadsheet['AG']
        else:
            big_spreadsheet['total-goals'] = big_spreadsheet['FTHG'] + big_spreadsheet['FTAG']
        big_df = big_df.append(big_spreadsheet, sort=False,ignore_index=True)
    return big_df

def calc_success_rate(df):
    # games won / total games played
    aa = df
    bb=aa.loc[(df['FTR']=='A','AwayTeam')].append(df.loc[(df['FTR']=='H','HomeTeam')],ignore_index=True)
    bb = pd.DataFrame(bb)
    bb['win-count'] = 1
    aa['c'] = 1
    win_count = bb.groupby(0).count()

    tmp =aa['AwayTeam']
    tmp = tmp.append(aa['HomeTeam'])
    tmp = pd.DataFrame(tmp)
    tmp['c'] = 1
    total_count = tmp.groupby(0).count()
        
    total_count['Team'] = total_count.index
    merged = pd.concat([win_count,total_count],axis=1,sort=False)
    merged = merged.fillna(0)
    merged['win-rate']=merged['win-count']/merged['c']
    merged['total-count'] = merged['c']
#    merged = merged.reset_index().dropna().sort_values('win-rate',ascending=False).reset_index()
    o = merged[['win-count','total-count','win-rate']]
    return o

def find_win_odds(df):
    home_wins=df.query('FTR=="H"')['HomeTeam']
    away_wins=df.query('FTR=="A"')['AwayTeam']
    draws = df.loc[df['FTR']=="D"]
    draws['win-team'] = 'Draw'
#    home_wins['win-team'] = home_wins['HomeTeam']
#    away_wins['win-team'] = away_wins['AwayTeam']
    win_team = pd.DataFrame()
    win_team['win-team'] =home_wins.append(away_wins.append(draws['win-team']))
    win_team['FTR'] = df['FTR']
    win_team['B365H'] = df['B365H']
    win_team['B365A'] = df['B365A']
    win_team['B365D'] = df['B365D']
    
    return win_team

big_df = load_all_data(spreadsheets)
data_state.text('Pre-processing')



season = st.selectbox(
    "Select season", list(big_df['season'].unique()),0
)

division = st.selectbox(
    "Select division", list(big_df['Div'].unique()),0
)


data = big_df.loc[big_df['Div'].isin( [division])]

chosen_year = int(season[5:])
data_state.text(chosen_year)
hist_data = data[data['s-year'] < chosen_year]

data =data[data['season'].isin([season])]
data=data.dropna(axis=1,how='all')


# If no team is selected calculate distribution of the total number of goals


## Head to head filter

#home_teams.insert(0,'None')

#away_teams.insert(0,'None')
## Minimum number of goals
head_to_head = st.sidebar.checkbox('Head to head comparison')
if head_to_head:
    location_specific = st.sidebar.checkbox('Location specific')
    all_teams = list(pd.array(list(data['HomeTeam'].unique()) + list(data['AwayTeam'].unique())).unique())
    all_teams.sort()
    if location_specific:        
        home_choice = st.sidebar.selectbox("Select home team", 
                                           all_teams,0
                                          )        
        away_choice = st.sidebar.selectbox("Select away team", 
                                           all_teams,1
                                          )
    else:
#        all_teams = list(data['HomeTeam'].unique()) + list(data['AwayTeam'].unique())
#        all_teams.sort()
        home_choice = st.sidebar.selectbox("Select team 1", 
                                           all_teams,0
                                          )
        away_choice = st.sidebar.selectbox("Select team 2", 
                                           all_teams,1
                                          )

    total_goals_filter = st.sidebar.checkbox('Filter by total goals')



if head_to_head and home_choice != away_choice:
    if location_specific:
        head_to_head_data = hist_data[(hist_data['HomeTeam']==home_choice) & (hist_data['AwayTeam']==away_choice)]
    else:
        head_to_head_data = hist_data[((hist_data['HomeTeam']==home_choice) & (hist_data['AwayTeam']==away_choice)) |
              ((hist_data['AwayTeam']==home_choice) & (hist_data['HomeTeam']==away_choice))  ]
    if total_goals_filter:
        # Minimum number of goals
        min_total = int(head_to_head_data['total-goals'].min())
        max_total = int(head_to_head_data['total-goals'].max())
        total_goals = st.sidebar.number_input("Select games with >= _ total goals", 
                                              min_value=min_total, 
                                              max_value=max_total,
                                              value=min_total)
        variant_filter_decision = st.sidebar.checkbox('Apply x/y filter i.e. select all games where at least x out of y games had > _ `total-goals`')
        
        if variant_filter_decision:
            variant_filter = st.sidebar.selectbox(
                        "x/y", ['1/1','1/2','2/2','2/3','3/3','3/4','4/4','4/5','5/5'],0
                        )
            
        
        head_to_head_data = head_to_head_data.loc[head_to_head_data['total-goals'] >= total_goals]
        
        
    data_state = st.text('Head to head data')
             
    max_goals = int(head_to_head_data['total-goals'].max())
    min_goals = int(head_to_head_data['total-goals'].min())
        
    hist_values = np.histogram(
        head_to_head_data['total-goals'],bins=np.linspace(min_goals,max_goals,max_goals-min_goals+2))[0]
    hist_values = hist_values/sum(hist_values)
    
#    st.write("### Raw data", head_to_head_data['total-goals'],min_goals,max_goals,
#                 hist_values,np.linspace(min_goals,max_goals,max_goals-min_goals+1))
                      
    hist_values = pd.DataFrame({'proportion':hist_values,'total-goals':np.linspace(min_goals,max_goals,max_goals-min_goals+1)})
    #st.title('Distribution between {} and {}'.format(chosen_year-1,hist_data['s-year'].min()))
    ##
    #st.bar_chart(hist_values)
    #
    line_chart = alt.Chart(hist_values).mark_bar().encode(
        alt.X('total-goals', title='Total Goals'),
        alt.Y('proportion', title='Proportion'),
    #    color='category:N'
    ).properties(
        title='Total goals based on the data from {} to {}'.format(chosen_year-1,head_to_head_data['s-year'].min()-1)
    )
    
    st.altair_chart(line_chart)
    st.write("### Raw data", head_to_head_data)
             
         
##if home_choice != 'None':
##    away_teams.remove(home_choice)
#
#
#away_choice = st.sidebar.selectbox("Select team 2", 
#                                      away_teams,0
#                                     )
##if away_choice != 'None':
##    home_teams.remove(away_choice)
#
#
#if home_choice != 'None':
#    data = data[data['HomeTeam'] == home_choice]
#
#columns = st.multiselect(
#    "Select columns", list(data.columns),['HomeTeam','AwayTeam','FTHG','FTAG','total-goals']
#)
#
#
#success_data = data
#odds_data = data
#data = data[columns]
#
#
#
#data_state = st.text('DataFrame shape {}'.format(data.shape))
#
#st.write("### Raw data", data.sort_index())
#         
#
## Success rate
#success_teams=calc_success_rate(success_data)
#st.write("### Success rates", success_teams)
#         
#                  
#         
## Select odds
#all_odds_list = ["B365H","B365D","B365A","BSH","BSD","BSA","BWH","BWD","BWA","GBH","GBD","GBA","IWH","IWD","IWA","LBH","LBD","LBA","PSH","PH","PSD","PD","PSA","PA","SOH","SOD","SOA","SBH","SBD","SBA","SJH","SJD","SJA","SYH","SYD","SYA","VCH","VCD","VCA","WHH","WHD","WHA"]
#available_odd = list(set(all_odds_list).intersection(list(big_df.columns)))
#
#odds_chosen = st.multiselect(
#    "Select odds", list(available_odd),['B365H',"B365D","B365A"]
#)
#
## Find average price for some team winning
#
## Calculate Proffit/Loss for all chosen odds
#
#odds_df = find_win_odds(odds_data)
#
#data_state = st.text('DataFrame shape {}'.format(hist_data.shape))
#st.write("### Profit/Loss", hist_data.head())
         
