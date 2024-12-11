import pandas as pd
import numpy as np
import sys


for t in range(int(sys.argv[1]),int(sys.argv[2])):

    data = pd.read_csv("./datav5_t{}/norm_vdata_full_column.csv".format(t))
    val = pd.read_csv("./datav5_t{}/norm_val_full_column.csv".format(t))
    raw_data = pd.read_csv("./datav5_t{}/raw_vdata_full_column.csv".format(t))
    raw_val = pd.read_csv("./datav5_t{}/raw_val_full_column.csv".format(t))
    std_data = pd.read_csv("./datav5_t{}/std_vdata_full_column.csv".format(t))
    std_val = pd.read_csv("./datav5_t{}/std_val_full_column.csv".format(t))

    for c in data.columns:
        if c.startswith('home_team_is_') or c.startswith('home_team_season_is_') or c.startswith('away_team_is_') or c.startswith('away_team_season_is_'):
            data = data.drop([c],axis = 1)
            val = val.drop([c],axis = 1)
            raw_data = raw_data.drop([c],axis = 1)
            raw_val = raw_val.drop([c],axis = 1)
            std_data = std_data.drop([c],axis = 1)
            std_val = std_val.drop([c],axis = 1)

    data.to_csv("./datav5_t{}/norm_vdata.csv".format(t), index=False)
    val.to_csv("./datav5_t{}/norm_val.csv".format(t), index=False)
    raw_data.to_csv("./datav5_t{}/raw_vdata.csv".format(t), index=False)
    raw_val.to_csv("./datav5_t{}/raw_val.csv".format(t), index=False)
    std_data.to_csv("./datav5_t{}/std_vdata.csv".format(t), index=False)
    std_val.to_csv("./datav5_t{}/std_val.csv".format(t), index=False)