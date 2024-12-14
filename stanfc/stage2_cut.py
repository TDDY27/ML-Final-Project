'''
問chatGPT:
This is my current code, i want to use BayesOptSearch to search best parameters (including those i haven't consider yet) for catboost, how can i use it
//貼上你本來的code
'''

import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
data = pd.read_csv('train_data.csv')

data = data.fillna({'is_night_game': 0.0, 'home_team_win': 0.0})
data['is_night_game'] = data['is_night_game'].astype(int)
data['home_team_win'] = data['home_team_win'].astype(int)
data['season'] = data['season'].fillna(0)

numerical_columns = data.select_dtypes(include=['number']).columns  # Select only numeric columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
data['season'] = data['season'].astype(int)

categorical_features = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']
data[categorical_features] = data[categorical_features].fillna("NAN")

data['date'] = pd.to_datetime(data['date'], errors='coerce', format='%Y-%m-%d')

# Drop rows with invalid dates if any
data = data.dropna(subset=['date'])

# Extract month and day for sorting
#data['year'] = data['date'].dt.year
#data['month'] = data['date'].dt.month
#data['day'] = data['date'].dt.day

# Sort by month and day
#data = data.sort_values(by=['date']).reset_index(drop=True)

# Split the first 80% as training and remaining 20% as testing
#cutoff = int(0.8 * len(data))  # 80% index cutoff
train_data = data[data['season'] != 2023]
test_data = data[data['season'] == 2023]

# Drop helper columns (if not needed later)
train_data = train_data
test_data = test_data

# Print the sizes to verify
print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")

# Define X and y for training and testing
to_drop = ['id', 'home_team_win', 'date', 'season', 'home_team_season', 'away_team_season']
X_train = train_data.drop(to_drop, axis=1)
y_train = train_data['home_team_win']

X_test = test_data.drop(to_drop, axis=1)
y_test = test_data['home_team_win']

X = data.drop(to_drop, axis=1)
y = data['home_team_win']

