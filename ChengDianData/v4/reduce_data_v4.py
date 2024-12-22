import pandas as pd
import numpy as np
import sys


for t in range(int(sys.argv[1]),int(sys.argv[2])):

    data = pd.read_csv("./datav4_t{}/norm_data_full_column.csv".format(t))
    test1 = pd.read_csv("./datav4_t{}/norm_test1_full_column.csv".format(t))
    test2 = pd.read_csv("./datav4_t{}/norm_test2_full_column.csv".format(t))
    raw_data = pd.read_csv("./datav4_t{}/raw_data_full_column.csv".format(t))
    raw_test1 = pd.read_csv("./datav4_t{}/raw_test1_full_column.csv".format(t))
    raw_test2 = pd.read_csv("./datav4_t{}/raw_test2_full_column.csv".format(t))
    std_data = pd.read_csv("./datav4_t{}/std_data_full_column.csv".format(t))
    std_test1 = pd.read_csv("./datav4_t{}/std_test1_full_column.csv".format(t))
    std_test2 = pd.read_csv("./datav4_t{}/std_test2_full_column.csv".format(t))

    for c in data.columns:
        if c.startswith('home_team_is_') or c.startswith('home_team_season_is_') or c.startswith('away_team_is_') or c.startswith('away_team_season_is_'):
            data = data.drop([c],axis = 1)
            test1 = test1.drop([c],axis = 1)
            test2 = test2.drop([c],axis = 1)
            raw_data = raw_data.drop([c],axis = 1)
            raw_test1 = raw_test1.drop([c],axis = 1)
            raw_test2 = raw_test2.drop([c],axis = 1)
            std_data = std_data.drop([c],axis = 1)
            std_test1 = std_test1.drop([c],axis = 1)
            std_test2 = std_test2.drop([c],axis = 1)

    data.to_csv("./datav4_t{}/norm_data.csv".format(t), index=False)
    test1.to_csv("./datav4_t{}/norm_test1.csv".format(t), index=False)
    test2.to_csv("./datav4_t{}/norm_test2.csv".format(t), index=False)
    raw_data.to_csv("./datav4_t{}/raw_data.csv".format(t), index=False)
    raw_test1.to_csv("./datav4_t{}/raw_test1.csv".format(t), index=False)
    raw_test2.to_csv("./datav4_t{}/raw_test2.csv".format(t), index=False)
    std_data.to_csv("./datav4_t{}/std_data.csv".format(t), index=False)
    std_test1.to_csv("./datav4_t{}/std_test1.csv".format(t), index=False)
    std_test2.to_csv("./datav4_t{}/std_test2.csv".format(t), index=False)