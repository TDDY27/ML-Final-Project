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
    test1 = pd.read_csv('test1.csv')
    test2 = pd.read_csv('test2.csv')

    data = data.sample(frac=1,random_state=int(sys.argv[1])).reset_index(drop=True)

    data_N = data.shape[0]
    test1_N = test1.shape[0]
    test2_N = test2.shape[0]

    win = data['home_team_win'].copy()

    name_set = set()
    season_set = set()
    team_season_set = set()

    for i in range(data_N):
        name_set.add(data.at[i, "home_team_abbr"])
        if not pd.isna(data.at[i, "season"]):
            season_set.add(int(data.at[i, "season"]))
        if not pd.isna(data.at[i, "home_team_season"]):
            team_season_set.add(data.at[i, "home_team_season"])
    for i in range(test2_N):
        name_set.add(test2.at[i, "home_team_abbr"])
        if not pd.isna(test2.at[i, "season"]):
            season_set.add(int(test2.at[i, "season"]))
        if not pd.isna(test2.at[i, "home_team_season"]):
            team_season_set.add(test2.at[i, "home_team_season"])

    name_list = list(name_set)
    name_list.sort()
    season_list = list(season_set)
    season_list.sort()
    team_season_list = list(team_season_set)
    team_season_list.sort()
    for name in name_list:
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['team_is_'+name])   
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['home_team_is_'+name])   
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['away_team_is_'+name])
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['team_is_'+name])
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['home_team_is_'+name])   
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['away_team_is_'+name])
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['team_is_'+name])
        test2 = pd.concat([test2,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['home_team_is_'+name])   
        test2 = pd.concat([test2,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['away_team_is_'+name])
        test2 = pd.concat([test2,new_column],axis = 1)
    for season in season_list:
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['season_is_'+str(season)])   
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['season_is_'+str(season)])   
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['season_is_'+str(season)])
        test2 = pd.concat([test2,new_column],axis = 1)
    for team_season in team_season_list:
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['team_season_is_'+team_season])
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['home_team_season_is_'+team_season])
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*data_N),columns = ['away_team_season_is_'+team_season])
        data = pd.concat([data,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['team_season_is_'+team_season])
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['home_team_season_is_'+team_season])
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test1_N),columns = ['away_team_season_is_'+team_season])
        test1 = pd.concat([test1,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['team_season_is_'+team_season])
        test2 = pd.concat([test2,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['home_team_season_is_'+team_season])
        test2 = pd.concat([test2,new_column],axis = 1)
        new_column = pd.DataFrame(pd.Series([0]*test2_N),columns = ['away_team_season_is_'+team_season])
        test2 = pd.concat([test2,new_column],axis = 1)

    new_column = pd.DataFrame(pd.Series([1]*data_N),columns = ['is_home_team'])   
    data = pd.concat([new_column,data],axis = 1)
    new_column = pd.DataFrame(pd.Series([1]*test1_N),columns = ['is_home_team'])   
    test1 = pd.concat([new_column,test1],axis = 1)
    new_column = pd.DataFrame(pd.Series([1]*test2_N),columns = ['is_home_team'])
    test2 = pd.concat([new_column,test2],axis = 1)

    for i in range(data_N):
        data.at[i, 'home_team_is_'+data.at[i, 'home_team_abbr']] = 1
        data.at[i, 'away_team_is_'+data.at[i, 'away_team_abbr']] = 1
        if not pd.isna(data.at[i, 'season']):
            data.at[i, 'season_is_'+str(int(data.at[i, 'season']))] = 1
        if not pd.isna(data.at[i, 'home_team_season']):
            data.at[i, 'home_team_season_is_'+data.at[i, 'home_team_season']] = 1
        if not pd.isna(data.at[i, 'away_team_season']):
            data.at[i, 'away_team_season_is_'+data.at[i, 'away_team_season']] = 1
    for i in range(test1_N):
        test1.at[i, 'home_team_is_'+test1.at[i, 'home_team_abbr']] = 1
        test1.at[i, 'away_team_is_'+test1.at[i, 'away_team_abbr']] = 1
        if not pd.isna(test1.at[i, 'season']):
            test1.at[i, 'season_is_'+str(int(test1.at[i, 'season']))] = 1
        if not pd.isna(test1.at[i, 'home_team_season']):
            test1.at[i, 'home_team_season_is_'+test1.at[i, 'home_team_season']] = 1
        if not pd.isna(test1.at[i, 'away_team_season']):
            test1.at[i, 'away_team_season_is_'+test1.at[i, 'away_team_season']] = 1
    for i in range(test2_N):
        test2.at[i, 'home_team_is_'+test2.at[i, 'home_team_abbr']] = 1
        test2.at[i, 'away_team_is_'+test2.at[i, 'away_team_abbr']] = 1
        if not pd.isna(test2.at[i, 'season']):
            test2.at[i, 'season_is_'+str(int(test2.at[i, 'season']))] = 1
        if not pd.isna(test2.at[i, 'home_team_season']):
            test2.at[i, 'home_team_season_is_'+test2.at[i, 'home_team_season']] = 1
        if not pd.isna(test2.at[i, 'away_team_season']):
            test2.at[i, 'away_team_season_is_'+test2.at[i, 'away_team_season']] = 1

    diff_list = [
        "team_rest",
        "pitcher_rest",
        "batting_RBI_10RA",
        "team_spread_mean",
        "batting_RBI_mean"
    ]

    for s in diff_list:
        data['diff_'+s] = data['home_'+s] - data['away_'+s]
        data = data.drop(['home_'+s],axis = 1)
        data = data.drop(['away_'+s],axis = 1)
        
        test1['diff_'+s] = test1['home_'+s] - test1['away_'+s]
        test1 = test1.drop(['home_'+s],axis = 1)
        test1 = test1.drop(['away_'+s],axis = 1)

        test2['diff_'+s] = test2['home_'+s] - test2['away_'+s]
        test2 = test2.drop(['home_'+s],axis = 1)
        test2 = test2.drop(['away_'+s],axis = 1)

    ratio_list = [
        "batting_batting_avg_10RA",
        "batting_onbase_perc_10RA",
        "batting_onbase_plus_slugging_10RA",
        "pitching_earned_run_avg_10RA",
        "pitching_SO_batters_faced_10RA",
        "pitching_H_batters_faced_10RA",
        "pitching_BB_batters_faced_10RA",
        "pitcher_earned_run_avg_10RA",
        "pitcher_SO_batters_faced_10RA",
        "pitcher_H_batters_faced_10RA",
        "pitcher_BB_batters_faced_10RA",
        "team_errors_mean",
        "batting_batting_avg_mean",
        "batting_onbase_perc_mean",
        "batting_onbase_plus_slugging_mean",
        "batting_wpa_bat_mean",
        "pitching_earned_run_avg_mean",
        "pitching_SO_batters_faced_mean",
        "pitching_H_batters_faced_mean",
        "pitching_BB_batters_faced_mean",
        "pitching_wpa_def_mean",
        "pitcher_earned_run_avg_mean",
        "pitcher_SO_batters_faced_mean",
        "pitcher_H_batters_faced_mean",
        "pitcher_BB_batters_faced_mean",
        "pitcher_wpa_def_mean"
    ]

    for s in ratio_list:
        data['ratio_'+s] = (data['home_'+s] * data['home_'+s].apply(abs)) - (data['away_'+s] * data['away_'+s].apply(abs))
        data['ratio_'+s] = data['ratio_'+s].apply(sq)
        data = data.drop(['home_'+s],axis = 1)
        data = data.drop(['away_'+s],axis = 1)
        
        test1['ratio_'+s] = (test1['home_'+s] * test1['home_'+s].apply(abs)) - (test1['away_'+s] * test1['away_'+s].apply(abs))
        test1['ratio_'+s] = test1['ratio_'+s].apply(sq)
        test1 = test1.drop(['home_'+s],axis = 1)
        test1 = test1.drop(['away_'+s],axis = 1)

        test2['ratio_'+s] = (test2['home_'+s] * test2['home_'+s].apply(abs)) - (test2['away_'+s] * test2['away_'+s].apply(abs))
        test2['ratio_'+s] = test2['ratio_'+s].apply(sq)
        test2 = test2.drop(['home_'+s],axis = 1)
        test2 = test2.drop(['away_'+s],axis = 1)

    drop_list = [
        "season",
        "home_team_win",
        "id",
        "date",
        "home_team_abbr",
        "away_team_abbr",
        "home_pitcher",
        "away_pitcher",
        "home_batting_leverage_index_avg_10RA",
        "away_batting_leverage_index_avg_10RA",
        "home_team_season",
        "away_team_season",
        "home_team_wins_mean",
        "away_team_wins_mean",
        "home_team_errors_std",
        "home_team_errors_skew",
        "away_team_errors_std",
        "away_team_errors_skew",
        "home_team_spread_std",
        "home_team_spread_skew",
        "away_team_spread_std",
        "away_team_spread_skew",
        "home_team_wins_std",
        "away_team_wins_std",
        "home_team_wins_skew",
        "away_team_wins_skew",
        "home_batting_leverage_index_avg_mean",
        "home_pitching_leverage_index_avg_mean",
        "home_pitcher_leverage_index_avg_mean",
        "home_batting_leverage_index_avg_std",
        "home_pitching_leverage_index_avg_std",
        "home_pitcher_leverage_index_avg_std",
        "home_batting_leverage_index_avg_skew",
        "home_pitching_leverage_index_avg_skew",
        "home_pitcher_leverage_index_avg_skew",
        "away_batting_leverage_index_avg_mean",
        "away_pitching_leverage_index_avg_mean",
        "away_pitcher_leverage_index_avg_mean",
        "away_batting_leverage_index_avg_std",
        "away_pitching_leverage_index_avg_std",
        "away_pitcher_leverage_index_avg_std",
        "away_batting_leverage_index_avg_skew",
        "away_pitching_leverage_index_avg_skew",
        "away_pitcher_leverage_index_avg_skew",
        "home_batting_batting_avg_std",
        "home_batting_onbase_perc_std",
        "home_batting_onbase_plus_slugging_std",
        "home_batting_wpa_bat_std",
        "home_batting_batting_avg_skew",
        "home_batting_onbase_perc_skew",
        "home_batting_onbase_plus_slugging_skew",
        "home_batting_wpa_bat_skew",
        "away_batting_batting_avg_std",
        "away_batting_onbase_perc_std",
        "away_batting_onbase_plus_slugging_std",
        "away_batting_wpa_bat_std",
        "away_batting_batting_avg_skew",
        "away_batting_onbase_perc_skew",
        "away_batting_onbase_plus_slugging_skew",
        "away_batting_wpa_bat_skew",
        "home_batting_RBI_std",
        "home_batting_RBI_skew",
        "away_batting_RBI_std",
        "away_batting_RBI_skew",
        "home_pitching_earned_run_avg_std",
        "home_pitching_SO_batters_faced_std",
        "home_pitching_H_batters_faced_std",
        "home_pitching_BB_batters_faced_std",
        "home_pitching_wpa_def_std",
        "home_pitcher_earned_run_avg_std",
        "home_pitcher_SO_batters_faced_std",
        "home_pitcher_H_batters_faced_std",
        "home_pitcher_BB_batters_faced_std",
        "home_pitcher_wpa_def_std",
        "home_pitching_earned_run_avg_skew",
        "home_pitching_SO_batters_faced_skew",
        "home_pitching_H_batters_faced_skew",
        "home_pitching_BB_batters_faced_skew",
        "home_pitching_wpa_def_skew",
        "home_pitcher_earned_run_avg_skew",
        "home_pitcher_SO_batters_faced_skew",
        "home_pitcher_H_batters_faced_skew",
        "home_pitcher_BB_batters_faced_skew",
        "home_pitcher_wpa_def_skew",
        "away_pitching_earned_run_avg_std",
        "away_pitching_SO_batters_faced_std",
        "away_pitching_H_batters_faced_std",
        "away_pitching_BB_batters_faced_std",
        "away_pitching_wpa_def_std",
        "away_pitcher_earned_run_avg_std",
        "away_pitcher_SO_batters_faced_std",
        "away_pitcher_H_batters_faced_std",
        "away_pitcher_BB_batters_faced_std",
        "away_pitcher_wpa_def_std",
        "away_pitching_earned_run_avg_skew",
        "away_pitching_SO_batters_faced_skew",
        "away_pitching_H_batters_faced_skew",
        "away_pitching_BB_batters_faced_skew",
        "away_pitching_wpa_def_skew",
        "away_pitcher_earned_run_avg_skew",
        "away_pitcher_SO_batters_faced_skew",
        "away_pitcher_H_batters_faced_skew",
        "away_pitcher_BB_batters_faced_skew",
        "away_pitcher_wpa_def_skew"
    ]

    for s in drop_list:
        data = data.drop([s],axis = 1)
        if s not in ["home_team_win","date"]:
            test1 = test1.drop([s],axis = 1)
            test2 = test2.drop([s],axis = 1)

    data.insert(loc=0, column='home_team_win', value=win)

    bool_list = [
        "home_team_win",
        "is_night_game"
    ]
    for s in bool_list:
        data[s] = data[s].apply(float)
        if s not in ["home_team_win"]:
            test1[s] = test1[s].apply(float)
            test2[s] = test2[s].apply(float)

    for c in test1.columns:
        if c not in ['home_team_win']:
            data[c] = data[c].fillna(data[c].mean())
            test1[c] = test1[c].fillna(data[c].mean())
            test2[c] = test2[c].fillna(data[c].mean())

    data_b = data.copy()
    
    data_b['home_team_win'] = 1 - data_b['home_team_win']
    data_b['is_home_team'] = 0
    
    for c in diff_list:
        data_b['diff_'+c] = -data_b['diff_'+c]
    for c in ratio_list:
        data_b['ratio_'+c] = -data_b['ratio_'+c]
    for name in name_list:
        data['team_is_'+name] = data['home_team_is_'+name].copy()
        data_b['team_is_'+name] = data_b['away_team_is_'+name].copy()
    for team_season in team_season_list:
        data['team_season_is_'+team_season] = data['home_team_season_is_'+team_season].copy()
        data_b['team_season_is_'+team_season] = data_b['away_team_season_is_'+team_season].copy()
    data = pd.concat([data,data_b],axis = 0,ignore_index=True)
    for name in name_list:
        data = data.drop(['home_team_is_'+name], axis = 1)
        data = data.drop(['away_team_is_'+name], axis = 1)
    for team_season in team_season_list:
        data = data.drop(['home_team_season_is_'+team_season], axis = 1)
        data = data.drop(['away_team_season_is_'+team_season], axis = 1)
    
    test1_b = test1.copy()

    for c in diff_list:
        test1_b['diff_'+c] = -test1_b['diff_'+c]
    for c in ratio_list:
        test1_b['ratio_'+c] = -test1_b['ratio_'+c]
    for name in name_list:
        test1['team_is_'+name] = test1['home_team_is_'+name].copy()
        test1_b['team_is_'+name] = test1_b['away_team_is_'+name].copy()
    for team_season in team_season_list:
        test1['team_season_is_'+team_season] = test1['home_team_season_is_'+team_season].copy()
        test1_b['team_season_is_'+team_season] = test1_b['away_team_season_is_'+team_season].copy()
    test1 = pd.concat([test1,test1_b],axis = 0)
    for name in name_list:
        test1 = test1.drop(['home_team_is_'+name], axis = 1)
        test1 = test1.drop(['away_team_is_'+name], axis = 1)
    for team_season in team_season_list:
        test1 = test1.drop(['home_team_season_is_'+team_season], axis = 1)
        test1 = test1.drop(['away_team_season_is_'+team_season], axis = 1)
    
    test2_b = test2.copy()

    for c in diff_list:
        test2_b['diff_'+c] = -test2_b['diff_'+c]
    for c in ratio_list:
        test2_b['ratio_'+c] = -test2_b['ratio_'+c]
    for name in name_list:
        test2['team_is_'+name] = test2['home_team_is_'+name].copy()
        test2_b['team_is_'+name] = test2_b['away_team_is_'+name].copy()
    for team_season in team_season_list:
        test2['team_season_is_'+team_season] = test2['home_team_season_is_'+team_season].copy()
        test2_b['team_season_is_'+team_season] = test2_b['away_team_season_is_'+team_season].copy()

    test2 = pd.concat([test2,test2_b],axis = 0)
    for name in name_list:
        test2 = test2.drop(['home_team_is_'+name], axis = 1)
        test2 = test2.drop(['away_team_is_'+name], axis = 1)
    for team_season in team_season_list:
        test2 = test2.drop(['home_team_season_is_'+team_season], axis = 1)
        test2 = test2.drop(['away_team_season_is_'+team_season], axis = 1)
    
    data.to_csv("./datav7_t{}/raw_data_full_column.csv".format(t), index=False)
    test1.to_csv("./datav7_t{}/raw_test1_full_column.csv".format(t), index=False)
    test2.to_csv("./datav7_t{}/raw_test2_full_column.csv".format(t), index=False)
    std_test1 = test1.copy()
    std_test2 = test2.copy()
    std_data = data.copy()

    for c in test1.columns:
        if c not in ['home_team_win'] and (data[c].max() - data[c].min()) != 0:
            test1[c] = (test1[c] - data[c].min()) / (data[c].max() - data[c].min())
            test2[c] = (test2[c] - data[c].min()) / (data[c].max() - data[c].min())
            data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())
    for c in test1.columns:
        if c not in ['home_team_win'] and std_data[c].std() != 0:
            std_test1[c] = (std_test1[c] - std_data[c].mean()) / std_data[c].std()
            std_test2[c] = (std_test2[c] - std_data[c].mean()) / std_data[c].std()
            std_data[c] = (std_data[c] - std_data[c].mean()) / std_data[c].std()

    data.to_csv("./datav7_t{}/norm_data_full_column.csv".format(t), index=False)
    test1.to_csv("./datav7_t{}/norm_test1_full_column.csv".format(t), index=False)
    test2.to_csv("./datav7_t{}/norm_test2_full_column.csv".format(t), index=False)
    std_data.to_csv("./datav7_t{}/std_data_full_column.csv".format(t), index=False)
    std_test1.to_csv("./datav7_t{}/std_test1_full_column.csv".format(t), index=False)
    std_test2.to_csv("./datav7_t{}/std_test2_full_column.csv".format(t), index=False)