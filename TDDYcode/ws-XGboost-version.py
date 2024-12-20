import pandas as pd  # For data manipulation and reading CSV files
import numpy as np  # For numerical computations
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.model_selection import cross_val_score  # Cross-validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


train_data = pd.read_csv("/home/tddy/ML/Final project/html-2024-fall-final-project-stage-1/train_data.csv")  
# Drop rows with missing values in the specified columns

#fill Nan
numerical_columns = train_data.select_dtypes(include=['number']).columns  # Select only numeric columns
train_data[numerical_columns] = train_data[numerical_columns].fillna(train_data[numerical_columns].mean())

# One-hot encode 'home_team_season' and 'away_team_season'
original_columns = train_data.columns.tolist()

#one hot encode
onehot_columns = ['home_team_season', 'away_team_season','home_team_abbr','away_team_abbr','season', ]
train_data = pd.get_dummies(train_data, columns=onehot_columns, drop_first=True)

# Save the new column names after one-hot encoding
new_columns = train_data.columns.tolist()

# Identify the newly added columns
dummy_columns = [col for col in new_columns if col not in original_columns]

print(dummy_columns)



# #fill train_data
# new_columns = pd.DataFrame({
#     'team_rest_diff': train_data['home_team_rest'] - train_data['away_team_rest'],
#     'pitcher_rest_diff': train_data['home_pitcher_rest'] - train_data['away_pitcher_rest'],
#     'batting_avg_diff': train_data['home_batting_batting_avg_10RA'] - train_data['away_batting_batting_avg_10RA'],
#     'pitching_ERA_diff': train_data['home_pitching_earned_run_avg_10RA'] - train_data['away_pitching_earned_run_avg_10RA'],
#     'onbase_perc' : train_data['home_batting_onbase_plus_slugging_10RA'] / train_data['away_batting_onbase_plus_slugging_10RA']
# })

# Concatenate the new columns to the original DataFrame
# train_data = pd.concat([train_data, new_columns], axis=1)
columns_to_drop = ['away_pitcher', 'home_pitcher', 'date','id']
train_data.drop(columns=columns_to_drop, inplace=True)
train_data['is_night_game'] = train_data['is_night_game'].apply(
    lambda x: np.random.choice([True, False]) if pd.isnull(x) else x
)

# Example of splitting the dataset by winning and losing teams
# Assuming 'home_team_win' is the column that indicates the outcome (1 = win, 0 = loss)
# winning_team_data = train_data[train_data['home_team_win'] == 1]
# losing_team_data = train_data[train_data['home_team_win'] == 0]
# numeric_columns = train_data.select_dtypes(include=['number']).columns
# # Loop through all columns and perform a t-test for each column
# count = 0
# add_colums=[]
# for column in numeric_columns:
#     if column != 'home_team_win':  # Skip the outcome column itself
#         # Perform t-test (assuming data is numeric)
#         t_stat, p_value = stats.ttest_ind(winning_team_data[column], losing_team_data[column], nan_policy='omit')
        
#         # If p-value is less than 0.05, print the result
#         if p_value < 0.05:
#             # print(f"Significant variable: {column} | p-value: {p_value}")
#             add_colums.append(column)

# print(train_data)
y = train_data['home_team_win']
X = train_data.drop(columns=['home_team_win'])

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=27)


def objective(trial):

    # Hyperparameter search space
    param = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 2,
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1, 5),
    }

    # Train model
    model = xgb.XGBClassifier(**param)
    model.fit(trainX, trainy)

    # Evaluate
    y_pred = model.predict(testX)
    accuracy = accuracy_score(testy, y_pred)
    return accuracy

# Optimize using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Display results
print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)
