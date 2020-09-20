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

@st.cache
def load_all_data(spreadsheets):
    # Silly. But it works..
    base_url = 'https://www.football-data.co.uk/'
    big_df = pd.DataFrame()
    all_keys = list(spreadsheets.keys())
    stop_season = '2014-2015'
#    stop_season = '2018-2019'
    pos = all_keys.index(stop_season)
    for c,key in enumerate(all_keys):
        print(key)
        
        if key == stop_season:
            break
        
        data_state.text('Loading season {} ... only {} left to go'.format(key,pos-c))
        url = base_url + spreadsheets[key]
        og_spreadsheet = pd.read_excel(url,None)
       
        big_spreadsheet = pd.concat(og_spreadsheet, ignore_index=True)
        # Convert date to datetime object
        #big_spreadsheet['Date'] = pd.to_datetime(big_spreadsheet['Date'])
        big_spreadsheet.loc[big_spreadsheet.index,'Date']  = pd.to_datetime(big_spreadsheet['Date'])
        #big_spreadsheet['season'] = key
        big_spreadsheet.loc[big_spreadsheet.index,'season']  = key
        #big_spreadsheet['s-year'] = key[5:]
        big_spreadsheet.loc[big_spreadsheet.index,'s-year']  = key[5:]
        #big_spreadsheet['s-year'] = big_spreadsheet['s-year'].astype(int)
        big_spreadsheet.loc[big_spreadsheet.index,'s-year']  = big_spreadsheet['s-year'].astype(int)
        if 'AG' in big_df.columns:
            #big_spreadsheet['total-goals'] = big_spreadsheet['HG'] + big_spreadsheet['AG']
            big_spreadsheet.loc[big_spreadsheet.index,'total-goals']  = big_spreadsheet['HG'] + big_spreadsheet['AG']
        else:
            #big_spreadsheet['total-goals'] = big_spreadsheet['FTHG'] + big_spreadsheet['FTAG']
            big_spreadsheet.loc[big_spreadsheet.index,'total-goals']  = big_spreadsheet['FTHG'] + big_spreadsheet['FTAG']
        big_df = big_df.append(big_spreadsheet, sort=False,ignore_index=True)
        big_df = big_df[big_df['total-goals'].isna()==False]
    return big_df.sort_values('Date',ascending=False)#.dropna(axis=0,how='any')

def prev_match(df,order_specific=False):
    small = df[['Date','HomeTeam','AwayTeam','total-goals']]
    small = small.dropna(how='all')
    if order_specific:
        small.loc[small.index,'hash'] = (small['HomeTeam'] + small['AwayTeam']).apply(hash)
    else:
        small.loc[small.index,'hash'] =  small['HomeTeam'].apply(hash) + small['AwayTeam'].apply(hash)
    
    return small.drop_duplicates(subset='hash', keep="first")

def prev_match_selection(df,order_specific=True,sel_type=None,total_goals=2.5):
    # Return list of pairs where certain condition was satisfied
    small = df[['Date','HomeTeam','AwayTeam','total-goals']]
    small = small.dropna(how='all')
    if order_specific:
        small.loc[small.index,'hash'] = (small['HomeTeam'] + small['AwayTeam']).apply(hash)
    else:
        small.loc[small.index,'hash'] =  small['HomeTeam'].apply(hash) + small['AwayTeam'].apply(hash)
    
    tmp = sel_type.split('/')
    games_played = int(tmp[1])
    min_goals = int(tmp[0])
        
    # Find the total matches played criteria
    grouped_matches = small.groupby('hash').head(games_played)
    # Only select matches where total-goals was satisfied
    filtered_matches = grouped_matches[grouped_matches['total-goals'].gt(total_goals)]
    # Count how many matches satisfied the total-goals criterion
    hash_sizes = filtered_matches.groupby('hash').size().reset_index(name='counts')
    # Only keep matches that satisfy the criterion
    good_hashes = hash_sizes[hash_sizes['counts'].ge(min_goals)]
    # Merge back to find Home and Away team names
    merged = pd.merge(small,good_hashes,left_on='hash',right_on='hash',copy=False)
    merged.loc[merged.index,'total-goals'] = np.ceil(total_goals)
    return merged[['HomeTeam','AwayTeam','total-goals','hash']].drop_duplicates()

def find_stats(test_df,decision_df,order_specific=True,stats_type='last-match',misc=None):
    # Add hashes appropriately
    if order_specific:
        test_df.loc[test_df.index,'hash'] = (test_df['HomeTeam']+test_df['AwayTeam']).apply(hash)
    else:
       test_df.loc[test_df.index,'hash'] = test_df['HomeTeam'].apply(hash)+test_df['AwayTeam'].apply(hash)
        
        
    o = {'accuracy':0,'data':None}
    
    if order_specific:
        # Match test_df with decision_df on hashes
        merged = pd.merge(test_df,decision_df,left_on='hash',right_on='hash',copy=False,suffixes=['_t','_d'])
        merged_full = merged
        merged = merged_full[['hash','total-goals_t','total-goals_d',#'HomeTeam_t','AwayTeam_t','HomeTeam_d','AwayTeam_d'
                         ]]
        merged.loc[merged.index,'correct']  = 0
        merged.loc[(
                    ((merged['total-goals_t']>2.5) &  (merged['total-goals_d']>2.5))
                    |
                    ((merged['total-goals_t']<2.5) &  (merged['total-goals_d']<2.5))                    
                    )
                        ,'correct'                   
                   ] = 1
        
        o['accuracy'] = merged['correct'].mean()
        if 'Date_t' in merged_full.keys():
            date_var = 'Date_t'
        else:
            date_var = 'Date'
        o['data'] = [merged_full[[date_var,'HomeTeam_t','AwayTeam_t','total-goals_t','total-goals_d']]]
        
    else:
        # This makes it harder, if more than one game, will have to update stats in between
        # Usually each season has two rounds for each team ??
        first_round = test_df.drop_duplicates(subset='hash', keep="last")
        second_round = test_df.drop(first_round.index)
        
#        st.write('first round',first_round)
#        st.write(first_round.shape)
#        st.write('second round',second_round)
#        st.write(second_round.shape)
#        st.write('test_df',test_df)
        
        # Workout decisions for the first round
        merged1 = pd.merge(first_round,decision_df,on='hash',copy=False,suffixes=['_t','_d'])
        merged1 = merged1.drop_duplicates(subset='hash', keep="last")
        merged1 = merged1.drop(columns=['HomeTeam_d','AwayTeam_d'])
#        st.write(merged1)
        
        res = merged1[['hash']]
        res['total-goals'] = merged1['total-goals_t']
        
        # Flag correct decision
        merged1.loc[merged1.index,'correct']  = 0
        merged1.loc[(
                    ((merged1['total-goals_t']>2.5) &  (merged1['total-goals_d']>2.5))
                    |
                    ((merged1['total-goals_t']<2.5) &  (merged1['total-goals_d']<2.5))                    
                    )
                        ,'correct'                   
                   ] = 1
        
#        st.write('first round choices',merged1[['HomeTeam_t','AwayTeam_t','total-goals_t','total-goals_d','correct']])
        
        # Update stats for second round
        if not second_round.empty:
            if stats_type == 'last-match':
                # Find total goals from previous play
                merged2 = pd.merge(second_round,res,left_on='hash',right_on='hash',copy=False,suffixes=['_t','_d'])
                merged2.loc[merged2.index,'correct']  = 0
                
                merged2.loc[(
                    ((merged2['total-goals_t']>2.5) &  (merged2['total-goals_d']>2.5))
                    |
                    ((merged2['total-goals_t']<2.5) &  (merged2['total-goals_d']<2.5))                    
                    )
                        ,'correct'                   
                   ] = 1
                
            elif stats_type == 'xytotal':
                if not misc is None:
                    x_y_type = misc['sel_type']
                    total_goals = misc['total_goals']
                    hist_data = misc['hist_data']
                    
                new_data_dirty = merged1.drop(['hash','correct'],axis=1)
                new_data = new_data_dirty.rename(columns={'HomeTeam_t':'HomeTeam','AwayTeam_t':'AwayTeam','total-goals_t':'total-goals'}).sort_values('Date',ascending=False)
                
                combined = new_data.append(hist_data,ignore_index=True)
                                                
                second_round_choices = prev_match_selection(combined,order_specific=order_specific,sel_type=x_y_type,total_goals=total_goals)
                merged2  = pd.merge(second_round,second_round_choices,on='hash',copy=False,suffixes=['_t','_d'])
               
                second_round_choices = second_round_choices[['hash','total-goals']].drop_duplicates()
                
#                st.write('second_round_choices',second_round_choices)
#                st.write(second_round_choices.shape)
                                
                 # Find total goals from previous play
                merged2 = pd.merge(second_round,second_round_choices,left_on='hash',right_on='hash',copy=False,suffixes=['_t','_d'])
                merged2.loc[merged2.index,'correct']  = 0
#                st.write(merged2[['HomeTeam','AwayTeam','total-goals_t','total-goals_d']])
                merged2.loc[(
                    ((merged2['total-goals_t']>2.5) &  (merged2['total-goals_d']>2.5))
                    |
                    ((merged2['total-goals_t']<2.5) &  (merged2['total-goals_d']<2.5))                    
                    )
                        ,'correct'                   
                   ] = 1
                               
            o['accuracy'] = np.array(list(merged2['correct'])+list(merged1['correct'])).mean()
            if 'Date_t' in merged1.keys():
                date_val = 'Date_t'
            else:
                date_val = 'Date'
            o['data'] = [merged1[[date_val,'HomeTeam_t','AwayTeam_t','total-goals_t','total-goals_d']],
                          merged2[['Date','HomeTeam','AwayTeam','total-goals_t','total-goals_d']]]
        else:
            o['accuracy'] = np.array(list(merged1['correct'])).mean()
            if 'Date_t' in merged1.keys():
                date_val = 'Date_t'
            else:
                date_val = 'Date'
            o['data'] = [merged1[[date_val,'HomeTeam_t','AwayTeam_t','total-goals_t','total-goals_d']]]
    return o

def calc_roi(season_odds,season_results,chosen_odds=None):
    # > 2.5 goals
    #BbAv>2.5 = Betbrain average over 2.5 goals
    #BbAv<2.5 = Betbrain average under 2.5 goals
    
    merged = pd.merge(season_odds,season_results,left_on=['Date','AwayTeam','HomeTeam'],right_on=['Date','AwayTeam','HomeTeam'],how='inner')

#    st.write('season_odds',season_odds.shape) 
#    st.write('season_results',season_results.shape) 
#
#    st.write('merged',merged)
#    st.write('merged',merged.shape)    
    
    clean = merged    
    
    bet_size = 1
    
    # Check that total-goals column was created
    # if not then go by the odds
    if 'total-goals_t' in clean.keys() and 'total-goals_d' in clean.keys():
        # add a flag to mark correctness
        clean.loc[clean.index,'correct>2.5'] = 0
        clean.loc[clean.index,'correct<2.5'] = 0
        clean.loc[clean.index,'correct'] = 0
        clean.loc[
                    ((clean['total-goals_t']>2.5) &  (clean['total-goals_d']>2.5))
                        ,'correct>2.5'                   
                   ] = 1
        clean.loc[
                    ((clean['total-goals_t']<2.5) &  (clean['total-goals_d']<2.5))
                        ,'correct<2.5'                   
                   ] = 1
        clean.loc[(clean['correct>2.5']==1) | (clean['correct<2.5']==1),'correct'] = 1
#        st.write(clean)
        
        broker_names = []
        won_sizes = []
        lost_sizes = []
        succ_rates = []
        avg_prices = []
        rois = []
        total_costs = []
        profits = []
        
        brokers = ['B365','P','GB','BbAv']
        avail_brokers = []
        # Lowest Broker for selection
        
        available_odds_gt = []
        available_odds_lt = []
        for b in brokers:
            b_str_gt = '{}>2.5'.format(b)
            b_str_lt = '{}<2.5'.format(b)
            if b_str_gt in clean.keys():
                available_odds_gt.append(b_str_gt)
                available_odds_lt.append(b_str_lt)
                avail_brokers.append(b)
        
        
        # Add new columns
        clean.loc[clean.index,'min>2.5']=clean[available_odds_gt].min(axis=1)
        clean.loc[clean.index,'max>2.5']=clean[available_odds_gt].max(axis=1)
        clean.loc[clean.index,'min<2.5']=clean[available_odds_lt].min(axis=1)
        clean.loc[clean.index,'max<2.5']=clean[available_odds_lt].max(axis=1)
        clean.loc[clean.index,'min-odds']=clean[available_odds_lt+available_odds_gt].min(axis=1)
        clean.loc[clean.index,'max-odds']=clean[list(available_odds_lt)+list(available_odds_gt)].max(axis=1)
        
        for c,b in enumerate(avail_brokers):
            broker = clean[[available_odds_gt[c],available_odds_lt[c],'correct','correct>2.5','correct<2.5']]
            broker = broker.dropna(axis=0,how='any')
            if broker.shape[0] > 0:            
                lost_size = broker['correct'].value_counts(dropna=False)[0]*bet_size
                
                correct_rows_gt = broker[broker['correct>2.5']==1]
                correct_rows_lt = broker[broker['correct<2.5']==1]

                won_size = (bet_size*(correct_rows_gt[available_odds_gt[c]]).sum(skipna=True) 
                            + bet_size*(correct_rows_lt[available_odds_lt[c]]).sum(skipna=True))
                profit = won_size - lost_size
                succ_rate = (correct_rows_gt.shape[0]+correct_rows_lt.shape[0])/(broker.shape[0])
                avg_price = np.array(list(correct_rows_gt[available_odds_gt[c]])+list(correct_rows_lt[available_odds_lt[c]])).mean()
                total_cost = bet_size*broker.shape[0]
                roi = profit/total_cost
                
                broker_names.append(b)
                won_sizes.append(won_size)
                lost_sizes.append(lost_size)
                succ_rates.append(succ_rate)
                avg_prices.append(avg_price)
                rois.append(roi)
                total_costs.append(total_cost)
                profits.append(profit)
        
        if 'B365C>2.5' in clean.keys():
            broker = clean[['B365C>2.5','B365C<2.5','correct','correct>2.5','correct<2.5']]
            broker = broker.dropna(axis=0,how='any')
            if broker.shape[0] > 0:
                lost_size = broker['correct'].value_counts(dropna=False)[0]*bet_size
                
                correct_rows_gt = broker[broker['correct>2.5']==1]
                correct_rows_lt = broker[broker['correct<2.5']==1]
                won_size = bet_size*(correct_rows_gt['B365C>2.5']+0).sum(skipna=True) + bet_size*(correct_rows_lt['B365C<2.5']+0).sum(skipna=True)
                
                profit = won_size - lost_size
                
                succ_rate = (correct_rows_gt.shape[0]+correct_rows_lt.shape[0])/(broker.shape[0])
                avg_price = np.array(list(correct_rows_gt['B365C>2.5'])+list(correct_rows_lt['B365C<2.5'])).mean()
                total_cost = bet_size*broker.shape[0]
                
                roi = profit/total_cost
                
                broker_names.append('Bet365Close')
                won_sizes.append(won_size)
                lost_sizes.append(lost_size)
                succ_rates.append(succ_rate)
                avg_prices.append(avg_price)
                rois.append(roi)
                total_costs.append(total_cost)
                profits.append(profit)
                
        if 'PC>2.5' in clean.keys():
            broker = clean[['PC>2.5','PC<2.5','correct','correct>2.5','correct<2.5']]
            broker = broker.dropna(axis=0,how='any')
            if broker.shape[0] > 0:
                lost_size = broker['correct'].value_counts(dropna=False)[0]*bet_size
                
                correct_rows_gt = broker[broker['correct>2.5']==1]
                correct_rows_lt = broker[broker['correct<2.5']==1]
                won_size = bet_size*(correct_rows_gt['PC>2.5']+0).sum(skipna=True) + bet_size*(correct_rows_lt['PC<2.5']+0).sum(skipna=True)
                
                profit = won_size - lost_size
                
                succ_rate = (correct_rows_gt.shape[0]+correct_rows_lt.shape[0])/(broker.shape[0])
                avg_price = np.array(list(correct_rows_gt['PC>2.5'])+list(correct_rows_lt['PC<2.5'])).mean()
                total_cost = bet_size*broker.shape[0]
                
                roi = profit/total_cost
                
                broker_names.append('PinnacleClose')
                won_sizes.append(won_size)
                lost_sizes.append(lost_size)
                succ_rates.append(succ_rate)
                avg_prices.append(avg_price)
                rois.append(roi)
                total_costs.append(total_cost)       
                profits.append(profit)
        
        # Select lowest broker
        broker = clean[['min>2.5','min<2.5','correct','correct>2.5','correct<2.5']]
        broker = broker.dropna(axis=0,how='any')
        if broker.shape[0] > 0:
            lost_size = broker['correct'].value_counts(dropna=False)[0]*bet_size
            
            correct_rows_gt = broker[broker['correct>2.5']==1]
            correct_rows_lt = broker[broker['correct<2.5']==1]
            won_size = bet_size*(correct_rows_gt['min>2.5']+0).sum(skipna=True) + bet_size*(correct_rows_lt['min<2.5']+0).sum(skipna=True)
            
            profit = won_size - lost_size
            
            succ_rate = (correct_rows_gt.shape[0]+correct_rows_lt.shape[0])/(broker.shape[0])
            avg_price = np.array(list(correct_rows_gt['min>2.5'])+list(correct_rows_lt['min<2.5'])).mean()
            total_cost = bet_size*broker.shape[0]
            
            roi = profit/total_cost
            
            broker_names.append('MinBroker')
            won_sizes.append(won_size)
            lost_sizes.append(lost_size)
            succ_rates.append(succ_rate)
            avg_prices.append(avg_price)
            rois.append(roi)
            total_costs.append(total_cost) 
            profits.append(profit)
            
        # Highest Broker
        broker = clean[['max>2.5','max<2.5','correct','correct>2.5','correct<2.5']]
        broker = broker.dropna(axis=0,how='any')
        if broker.shape[0] > 0:
            lost_size = broker['correct'].value_counts(dropna=False)[0]*bet_size
            
            correct_rows_gt = broker[broker['correct>2.5']==1]
            correct_rows_lt = broker[broker['correct<2.5']==1]
            won_size = bet_size*(correct_rows_gt['max>2.5']+0).sum(skipna=True) + bet_size*(correct_rows_lt['max<2.5']+0).sum(skipna=True)
            
            profit = won_size - lost_size
            
            succ_rate = (correct_rows_gt.shape[0]+correct_rows_lt.shape[0])/(broker.shape[0])
            avg_price = np.array(list(correct_rows_gt['max>2.5'])+list(correct_rows_lt['max<2.5'])).mean()
            total_cost = bet_size*broker.shape[0]
            
            roi = profit/total_cost 
            
            broker_names.append('MaxBroker')
            won_sizes.append(won_size)
            lost_sizes.append(lost_size)
            succ_rates.append(succ_rate)
            avg_prices.append(avg_price)
            rois.append(roi)
            total_costs.append(total_cost) 
            profits.append(profit)
        
            
        output_table = pd.DataFrame({'broker-name':broker_names,
                                     'won-size':won_sizes,
                                     'profit':profits,
#                                     'loss':lost_sizes,
                                     'succ-rate':succ_rates,
                                     'avg-price':avg_prices,
                                     'roi':rois,
                                     'total-cost':total_costs
                                     })
        st.write('### Selected odds',clean)
        st.write('### Results',output_table)
    else:
        #TODO: Calculate results based on odds (highest and lowest)
        pass
    
def filter_teams(df,chosen_n=5,filter_type='TotalAll'):
    # Select only last season data
    last_season = df[df['s-year']==df['s-year'].max()]
    # Rank = goals/num_games
    if filter_type == 'Total goals Home+Away':
        # Rank teams by total scored goals
        try:
            home = hist_data[['HomeTeam','FTHG']].rename(columns={'HomeTeam':'Team','FTHG':'Goals'})
        except:
            home = hist_data[['HomeTeam','HG']].rename(columns={'HomeTeam':'Team','HG':'Goals'})
        try:
            away = hist_data[['AwayTeam','FTAG']].rename(columns={'AwayTeam':'Team','FTAG':'Goals'})
        except:
            away = hist_data[['AwayTeam','AG']].rename(columns={'AwayTeam':'Team','AG':'Goals'})
        teams = home.append(away)
        
        goals_by_teams = teams[['Team','Goals']].groupby('Team').sum()
        games_by_teams = teams[['Team','Goals']].groupby('Team').count()
        rank = (goals_by_teams/games_by_teams).sort_values('Goals',ascending=False).head(chosen_n).index
        
        home_teams = pd.DataFrame(rank)
        merge_home = pd.merge(df,home_teams,left_on='HomeTeam',right_on='Team',how='inner')
        merge_away = pd.merge(df,home_teams,left_on='AwayTeam',right_on='Team',how='inner')
        
        merge = merge_home.append(merge_away).reset_index()
#        st.write(merge)
        return merge
    elif filter_type == 'Total goals Home':
        # Rank teams on total goals when was at home
        try:
            goals_by_teams = last_season[['HomeTeam','FTHG']].groupby('HomeTeam').sum()
            games_by_teams = last_season[['HomeTeam','FTHG']].groupby('HomeTeam').count()
            rank = (goals_by_teams/games_by_teams).sort_values('FTHG',ascending=False).head(chosen_n)
        except:
            goals_by_teams = last_season[['HomeTeam','HG']].groupby('HomeTeam').sum()
            games_by_teams = last_season[['HomeTeam','HG']].groupby('HomeTeam').count()
            rank = (goals_by_teams/games_by_teams).sort_values('HG',ascending=False).head(chosen_n).index
        home_teams = pd.DataFrame(rank)
        merge = pd.merge(df,home_teams,left_on='HomeTeam',right_on='HomeTeam',how='inner')
        return merge
    elif filter_type == 'Total goals Away':
#         Rank teams on total goals when was away
        try:
            goals_by_teams = last_season[['AwayTeam','FTAG']].groupby('AwayTeam').sum()
            games_by_teams = last_season[['AwayTeam','FTAG']].groupby('AwayTeam').count()
            rank = (goals_by_teams/games_by_teams).sort_values('FTAG',ascending=False).head(chosen_n)
        except:
            goals_by_teams = last_season[['AwayTeam','AG']].groupby('AwayTeam').sum()
            games_by_teams = last_season[['AwayTeam','AG']].groupby('AwayTeam').count()
            rank = (goals_by_teams/games_by_teams).sort_values('HG',ascending=False).head(chosen_n).index
        away_teams = pd.DataFrame(rank)
        merge = pd.merge(df,away_teams,left_on='AwayTeam',right_on='AwayTeam',how='inner')
        return merge

spreadsheets = find_all_spreadsheets()
data_state = st.text('')

data_state.text('Pre-processing')
big_df = load_all_data(spreadsheets)
data_state.text('')

season = st.selectbox(
    "Select season", list(big_df['season'].unique()),1
)

division = st.selectbox(
    "Select division", list(big_df['Div'].sort_values().unique()),0
)


order_specific = st.sidebar.checkbox('Order specific',1)

# Select by exact total number of goals
top_n_selection = st.sidebar.checkbox('Top n teams from the previous season',0)

st.markdown("""Select a type of Head to head. Two options are available.""")
st.markdown("1) Total goals from previous fixture looks at the previous total number of goals in the previous identical match")
st.markdown("2) x/y matching with total goals only selects matches where at least x out of y last matches had at least (however many)`total goals'")
st.markdown("More filters available in pannel on the left")

# Find previous total for all pairs
total_type = st.selectbox(
    "Type of `Head to Head'", ['None','Total goals from previous fixture',"x/y & `total goals' criterion"],0
)


current_year = int(season[5:])
division_data = big_df.loc[big_df['Div'] == division]
current_data = division_data.loc[big_df['s-year']==current_year]
hist_data = division_data.loc[(big_df['s-year'] < current_year)]

if top_n_selection:
    rank_type = st.sidebar.selectbox(
            "Rank teams by", ['Total goals Home+Away','Total goals Home','Total goals Away'],0
            )
    n = st.sidebar.number_input('Number of top teams selected', 
          min_value=1, 
          max_value=len(current_data['HomeTeam'].unique()+current_data['AwayTeam'].unique()),
          value=5)
    # Filter teams
    hist_data = filter_teams(hist_data,chosen_n=n,filter_type=rank_type)

test_data = None
stats_type = None
misc = None

if total_type == 'None':
    pass
elif total_type == 'Total goals from previous fixture':
    test_data = prev_match(hist_data,order_specific=order_specific)
    stats_type = 'last-match'
elif total_type == "x/y & `total goals' criterion":
    x_y_type = st.selectbox(
            "Select x/y", ['1/1','1/2','2/2','2/3','3/3','3/4','4/4','4/5','5/5'],5
            )
    total_goals = st.selectbox(
            "Select `total goals'", np.linspace(0.5,8.5,9),2
            )
    test_data = prev_match_selection(hist_data,order_specific=order_specific,sel_type=x_y_type,total_goals=total_goals)
    stats_type = 'xytotal'
    misc = {'sel_type':x_y_type,'total_goals':total_goals,'hist_data':hist_data}


# Workout how many matches were won with given filters
if total_type != 'None':
    temp = find_stats(current_data,test_data,order_specific=order_specific,stats_type=stats_type,misc=misc)
else:
    temp = {'data':[]}


if len(temp['data']) == 1:
    out_data = temp['data'][0].rename(columns={'HomeTeam_t':'HomeTeam',
                                               'AwayTeam_t':'AwayTeam',
#                                               'total-goals_t':'total-goals',
#                                               'total-goals_d':'total-goals',
                                               'Date_t':'Date',
                                               'Date_d':'Date'
                               })
    
#    st.write('## Selection',out_data)
elif len(temp['data']) == 2:
    out_data1 = temp['data'][0].rename(columns={'HomeTeam_t':'HomeTeam',
                                               'AwayTeam_t':'AwayTeam',
#                                               'total-goals_t':'total-goals',
#                                               'total-goals_d':'total-goals',
                                               'Date_t':'Date',
                                               'Date_d':'Date'
                               })
    out_data2 = temp['data'][1].rename(columns={'HomeTeam_t':'HomeTeam',
                                               'AwayTeam_t':'AwayTeam',
#                                               'total-goals_t':'total-goals',
#                                               'total-goals_d':'total-goals',
                                               'Date_t':'Date',
                                               'Date_d':'Date'
                               })
    out_data = out_data1.append(out_data2,ignore_index=True)
#    st.write('## Selection',out_data)

if total_type != 'None':
    calc_roi(current_data,out_data)
else:
    #TODO: Choose best matches based on odds
    pass
