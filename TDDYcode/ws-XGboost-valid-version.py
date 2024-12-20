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
import optuna

# Load dataset
# data = pd.read_csv('/tmp2/b12902127/ML-Final-Project/html-2024-fall-final-project-stage-1/train_data.csv')
data=pd.read_csv("/home/tddy/ML/Final project/html-2024-fall-final-project-stage-1/train_data.csv")  
testdata=pd.read_csv("/home/tddy/ML/Final project/html-2024-fall-final-project-stage-1/html2024-fall-final-project-stage-2/2024_test_data.csv")


data = data.fillna({'is_night_game': 0.0, 'home_team_win': 0.0})
data['is_night_game'] = data['is_night_game'].astype(int)
data['home_team_win'] = data['home_team_win'].astype(int)
data['season'] = data['season'].fillna(0)

numerical_columns = data.select_dtypes(include=['number']).columns  # Select only numeric columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
data['season'] = data['season'].astype(int)

categorical_features = ['season', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 
                        'home_team_season', 'away_team_season']
data[categorical_features] = data[categorical_features].fillna("NAN")

to_drop = ['id', 'home_team_win', 'date']
'''


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

data['date'] = pd.to_datetime(data['date'], errors='coerce', format='%Y-%m-%d')

# Drop rows with invalid dates if any
data = data.dropna(subset=['date'])

# Extract month and day for sorting
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Sort by month and day
data = data.sort_values(by=['month', 'day']).reset_index(drop=True)

# Drop rows with missing values in the specified columns

#fill Nan
numerical_columns = data.select_dtypes(include=['number']).columns  # Select only numeric columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# One-hot encode 'home_team_season' and 'away_team_season'
original_columns = data.columns.tolist()

#one hot encode
onehot_columns = ['home_team_season', 'away_team_season','home_team_abbr','away_team_abbr','season', ]
data = pd.get_dummies(data, columns=onehot_columns, drop_first=True)

# Save the new column names after one-hot encoding
new_columns = data.columns.tolist()

# Identify the newly added columns
dummy_columns = [col for col in new_columns if col not in original_columns]

# print(dummy_columns)

columns_to_drop = ['away_pitcher', 'home_pitcher', 'date','id']
data.drop(columns=columns_to_drop, inplace=True)
data['is_night_game'] = data['is_night_game'].apply(
    lambda x: np.random.choice([True, False]) if pd.isnull(x) else x
)

# test data processing
testdata = testdata.fillna({'is_night_game': 0.0, 'home_team_win': 0.0})
testdata['is_night_game'] = testdata['is_night_game'].astype(int)
testdata['home_team_win'] = testdata['home_team_win'].astype(int)
testdata['season'] = testdata['season'].fillna(0)
numerical_columns = testdata.select_dtypes(include=['number']).columns  # Select only numeric columns
testdata[numerical_columns] = testdata[numerical_columns].fillna(testdata[numerical_columns].mean())
testdata['season'] = testdata['season'].astype(int)

categorical_features = ['season', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 
                        'home_team_season', 'away_team_season']
testdata[categorical_features] = testdata[categorical_features].fillna("NAN")

to_drop = ['id', 'home_team_win', 'date']

# One-hot encode 'home_team_season' and 'away_team_season'
original_columns = testdata.columns.tolist()

#one hot encode
onehot_columns = ['home_team_season', 'away_team_season','home_team_abbr','away_team_abbr','season', ]
testdata = pd.get_dummies(testdata, columns=onehot_columns, drop_first=True)

# Save the new column names after one-hot encoding
new_columns = testdata.columns.tolist()

# Identify the newly added columns
dummy_columns = [col for col in new_columns if col not in original_columns]

# print(dummy_columns)
columns_to_drop = ['away_pitcher', 'home_pitcher', 'date','id']
testdata.drop(columns=columns_to_drop, inplace=True)
testdata['is_night_game'] = testdata['is_night_game'].apply(
    lambda x: np.random.choice([True, False]) if pd.isnull(x) else x
)


# Split the first 80% as training and remaining 20% as testing
cutoff = int(0.8 * len(data))  # 80% index cutoff
train_data = data.iloc[:cutoff]
test_data = data.iloc[cutoff:]

# Drop helper columns (if not needed later)
train_data = train_data.drop(columns=['month', 'day'])
test_data = test_data.drop(columns=['month', 'day'])
data=data.drop(columns=['month', 'day'])
# Print the sizes to verify
# print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")

# Define X and y for training and testing
X_train = train_data.drop(columns=['home_team_win'])
y_train = train_data['home_team_win']

X_test = test_data.drop(columns=['home_team_win'])
y_test = test_data['home_team_win']

X = data.drop(columns=['home_team_win'])
y = data['home_team_win']

param = {
    "verbosity": 0,
    "objective": "multi:softmax",
    "num_class": 2,
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "learning_rate": 0.06878298628903712,
    "n_estimators": 53 ,
    "subsample": 0.31490312644463636,
    "colsample_bytree": 0.6050549554091992,
    "gamma": 5.0214264964575595,
    "lambda": 9.517586328422444
}

# Train model
model = xgb.XGBClassifier(**param)
model.fit(X, y)

# Evaluate
y_pred = model.predict(testdata)
# print(y_pred)

# data_new = pd.read_csv('same_season_test_data.csv')
# data_new = data_new.fillna({'is_night_game': 0.0})
# data_new['is_night_game'] = data_new['is_night_game'].astype(int)
# data_new['season'] = data_new['season'].fillna(0)

# numerical_columns = data_new.select_dtypes(include=['number']).columns  # Select only numeric columns
# data_new[numerical_columns] = data_new[numerical_columns].fillna(data_new[numerical_columns].mean())
# data_new[categorical_features] = data_new[categorical_features].fillna("NAN")
# data_new['season'] = data_new['season'].astype(int)

# X_new = data_new.drop(['id'], axis=1)



# def objective(trial):

#     # Hyperparameter search space
#     param = {
#         "verbosity": 0,
#         "objective": "multi:softmax",
#         "num_class": 2,
#         "eval_metric": "mlogloss",
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "lambda": trial.suggest_float("lambda", 1, 5),
#     }

#     # Train model
#     model = xgb.XGBClassifier(**param)
#     model.fit(X_train, y_train)

#     # Evaluate
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy

# # Optimize using Optuna
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# # Display results
# print("Best parameters:", study.best_params)
# print("Best accuracy:", study.best_value)
