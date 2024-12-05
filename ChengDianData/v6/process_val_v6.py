import pandas as pd
import numpy as np
import math
import sys

def sq(x):
    return sign(x) * math.sqrt(abs(x))
def sign(x):
    if x > 0:
        return 1
    else:
        return -1

argc = len(sys.argv)

for t in range(int(sys.argv[1]),int(sys.argv[2])):

    data = pd.read_csv('data.csv')

    data = data.sample(frac=1,random_state=t).reset_index(drop=True)

    data_N = data.shape[0]

    win = data['home_team_win']

    name_set = set()
    season_set = set()
    team_season_set = set()

    for i in range(data_N):
        name_set.add(data.at[i, "home_team_abbr"])
        if not pd.isna(data.at[i, "season"]):
            season_set.add(int(data.at[i, "season"]))
        if not pd.isna(data.at[i, "home_team_season"]):
            team_season_set.add(data.at[i, "home_team_season"])
    
    name_list = list(name_set)
    name_list.sort()
    season_list = list(season_set)
    season_list.sort()
    team_season_list = list(team_season_set)
    team_season_list.sort()

    for name in name_list:
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['home_team_is_'+name])   
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['away_team_is_'+name])   
        data = pd.concat([data,new_column],axis = 1)
    for season in season_list:
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['season_is_'+str(season)])   
        data = pd.concat([data,new_column],axis = 1)
    for team_season in team_season_list:
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['home_team_season_is_'+team_season])   
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['away_team_season_is_'+team_season])   
        data = pd.concat([data,new_column],axis = 1)
    
    for i in range(data_N):
        data.at[i, 'home_team_is_'+data.at[i, 'home_team_abbr']] = 1
        data.at[i, 'away_team_is_'+data.at[i, 'away_team_abbr']] = 1
        if not pd.isna(data.at[i, 'season']):
            data.at[i, 'season_is_'+str(int(data.at[i, 'season']))] = 1
        if not pd.isna(data.at[i, 'home_team_season']):
            data.at[i, 'home_team_season_is_'+data.at[i, 'home_team_season']] = 1
        if not pd.isna(data.at[i, 'away_team_season']):
            data.at[i, 'away_team_season_is_'+data.at[i, 'away_team_season']] = 1

    data.insert(loc=0, column='home_team_win', value=win)

    bool_list = [
        "home_team_win",
        "is_night_game"
    ]
    for s in bool_list:
        data[s] = data[s].apply(float)

    val = data.loc[range((9*data_N)//10,data_N)].reset_index(drop=True).copy()
    data = data.loc[range(0,(9*data_N)//10)].copy()

    for c in val.columns:
        if c not in ['home_team_win']:
            data[c] = data[c].fillna(data[c].mean())
            val[c] = val[c].fillna(data[c].mean())
    data.to_csv("./datav6_t{}/raw_vdata_full_column.csv".format(t), index=False)
    val.to_csv("./datav6_t{}/raw_val_full_column.csv".format(t), index=False)

    std_val = val.copy()
    std_data = data.copy()

    for c in val.columns:
        if c not in ['home_team_win']:
            val[c] = (val[c] - data[c].min()) / (data[c].max() - data[c].min())
            data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())
    for c in val.columns:
        if c not in ['home_team_win']:
            std_val[c] = (std_val[c] - std_data[c].mean()) / std_data[c].std()
            std_data[c] = (std_data[c] - std_data[c].mean()) / std_data[c].std()
    for name in name_list:
        val['home_team_is_'+name] /= 2
        val['away_team_is_'+name] /= 2
        data['home_team_is_'+name] /= 2
        data['away_team_is_'+name] /= 2
        std_val['home_team_is_'+name] /= 2
        std_val['away_team_is_'+name] /= 2
        std_data['home_team_is_'+name] /= 2
        std_data['away_team_is_'+name] /= 2
    for season in season_list:
        val['season_is_'+str(season)] = 0
        data['season_is_'+str(season)] = 0
        std_val['season_is_'+str(season)] = 0
        std_data['season_is_'+str(season)] = 0
    for team_season in team_season_list:
        val['home_team_season_is_'+team_season] /= 2
        val['away_team_season_is_'+team_season] /= 2
        data['home_team_season_is_'+team_season] /= 2
        data['away_team_season_is_'+team_season] /= 2
        std_val['home_team_season_is_'+team_season] /= 2
        std_val['away_team_season_is_'+team_season] /= 2
        std_data['home_team_season_is_'+team_season] /= 2
        std_data['away_team_season_is_'+team_season] /= 2
    data.to_csv("./datav6_t{}/norm_vdata_full_column.csv".format(t), index=False)
    val.to_csv("./datav6_t{}/norm_val_full_column.csv".format(t), index=False)
    std_data.to_csv("./datav6_t{}/std_vdata_full_column.csv".format(t), index=False)
    std_val.to_csv("./datav6_t{}/std_val_full_column.csv".format(t), index=False)