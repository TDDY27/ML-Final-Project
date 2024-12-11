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

categorical_features = ['season', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 
                        'home_team_season', 'away_team_season']
data[categorical_features] = data[categorical_features].fillna("NAN")

to_drop = ['id', 'home_team_win', 'date']
X = data.drop(to_drop, axis=1)
y = data['home_team_win']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define objective function for Optuna
def objective(trial):
    # Suggest parameters
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10),
        "loss_function": "Logloss",
        "cat_features": categorical_features,
        "verbose": 0,
    }
    
    # Train model
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Optimize hyperparameters with Optuna
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=50)

# Best parameters
#print("Best parameters:", study.best_params)

# Train final model with best parameters
#best_params = study.best_params
best_params = {'iterations': 650,
                'depth': 4,
                'learning_rate': 0.011806668800701858, 
                'l2_leaf_reg': 5.077749635687785, 
                'bagging_temperature': 0.9868714879223595, 
                'random_strength': 2.556410701326167}
model = CatBoostClassifier(**best_params, loss_function='Logloss', cat_features=categorical_features, verbose=100)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)

# Save predictions
data_new = pd.read_csv('same_season_test_data.csv')
data_new = data_new.fillna({'is_night_game': 0.0})
data_new['is_night_game'] = data_new['is_night_game'].astype(int)
data_new['season'] = data_new['season'].fillna(0)

numerical_columns = data_new.select_dtypes(include=['number']).columns  # Select only numeric columns
data_new[numerical_columns] = data_new[numerical_columns].fillna(data_new[numerical_columns].mean())
data_new[categorical_features] = data_new[categorical_features].fillna("NAN")
data_new['season'] = data_new['season'].astype(int)

X_new = data_new.drop(['id'], axis=1)
y_pred_new = model.predict(X_new)

# Convert predictions to DataFrame
df = pd.DataFrame({
    'id': range(len(y_pred_new)),
    'home_team_win': y_pred_new
})
df.to_csv('predictions_CAT_VAL.csv', index=False)
