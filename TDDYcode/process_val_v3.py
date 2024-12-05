import csv
import pandas as pd
import numpy as np
import math
import sys
import subprocess

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

    for i in range(data_N):
        name_set.add(data.at[i, "home_team_abbr"])

    name_list = list(name_set)
    name_list.sort()

    for name in name_list:
        data['home_team_is_'+name] = 0
        data['away_team_is_'+name] = 0

    for i in range(data_N):
        data.at[i, 'home_team_is_'+data.at[i, 'home_team_abbr']] = 1
        data.at[i, 'away_team_is_'+data.at[i, 'away_team_abbr']] = 1

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

    data.insert(loc=0, column='home_team_win', value=win)

    bool_list = [
        "home_team_win",
        "is_night_game"
    ]
    for s in bool_list:
        data[s] = data[s].apply(float)

    val = data.loc[range((9*data_N)//10,data_N)].reset_index(drop=True)
    data = data.loc[range(0,(9*data_N)//10)]

    nc_data = data
    nc_val = val

    for name in name_list:
        nc_val = nc_val.drop(['home_team_is_'+name],axis = 1)
        nc_val = nc_val.drop(['away_team_is_'+name],axis = 1)
        nc_data = nc_data.drop(['home_team_is_'+name],axis = 1)
        nc_data = nc_data.drop(['away_team_is_'+name],axis = 1)

    for c in val.columns:
        data[c] = data[c].fillna(data[c].mean())
        val[c] = val[c].fillna(data[c].mean())
    data.to_csv("./datac3_t{}/raw_vdata_full_column.csv".format(t), index=False)
    val.to_csv("./datac3_t{}/raw_val_full_column.csv".format(t), index=False)

    for c in nc_val.columns:
        nc_data[c] = nc_data[c].fillna(nc_data[c].mean())
        nc_val[c] = nc_val[c].fillna(nc_val[c].mean())
    nc_data.to_csv("./datac3_t{}/raw_vdata.csv".format(t), index=False)
    nc_val.to_csv("./datac3_t{}/raw_val.csv".format(t), index=False)

    for c in val.columns:
        val[c] = (val[c] - data[c].min()) / (data[c].max() - data[c].min())
        data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())
    for name in name_list:
        val['home_team_is_'+name] /= 1000
        val['away_team_is_'+name] /= 1000
        data['home_team_is_'+name] /= 1000
        data['away_team_is_'+name] /= 1000
    data.to_csv("./datac3_t{}/norm_vdata_full_column.csv".format(t), index=False)
    val.to_csv("./datac3_t{}/norm_val_full_column.csv".format(t), index=False)

    for c in nc_val.columns:
        nc_val[c] = (nc_val[c] - nc_data[c].min()) / (nc_data[c].max() - nc_data[c].min())
        nc_data[c] = (nc_data[c] - nc_data[c].min()) / (nc_data[c].max() - nc_data[c].min())
    nc_data.to_csv("./datac3_t{}/norm_vdata.csv".format(t), index=False)
    nc_val.to_csv("./datac3_t{}/norm_val.csv".format(t), index=False)