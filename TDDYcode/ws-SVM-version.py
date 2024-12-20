import pandas as pd  # For data manipulation and reading CSV files
import numpy as np  # For numerical computations
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.model_selection import cross_val_score  # Cross-validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import optuna


train_data = pd.read_csv("/tmp2/b12902127/ML-Final-Progect/html-2024-fall-final-project-stage-1/train_data.csv")  
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

def thebest(trial, trainX, trainy, testX, testy) : 
    C = trial.suggest_loguniform('C', 0.1, 100.0)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    n_estimators = trial.suggest_int('n_estimators', 10, 50)
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0)
    max_features = trial.suggest_float('max_features', 0.3, 1.0)


    rbf_base_estimator = SVC(kernel='rbf', probability=True, C=C, gamma=gamma, random_state=27)
    bagging_rbf_svm = BaggingClassifier(
        estimator=SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=27),
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,
        bootstrap_features=True,
        random_state=27
    )

    rbf_base_estimator = SVC(kernel='linear', probability=True, C=C, gamma=gamma, random_state=27)
    bagging_linear_svm = BaggingClassifier(
        estimator=SVC(kernel='linear', C=10.0, gamma='auto',random_state=27),
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,
        bootstrap_features=True,
        random_state=27
    )
    # bagging_poly_svm = BaggingClassifier(
    #     base_estimator=SVC(kernel='poly', degree=2, probability=True),
    #     n_estimators=10,
    #     max_samples=0.8,
    #     max_features=0.8,
    #     bootstrap=True,
    #     bootstrap_features=True,
    #     random_state=42
    # )

    bagging_linear_svm.fit(trainX,trainy)
    bagging_rbf_svm.fit(trainX,trainy)

    linear_pred=bagging_linear_svm.predict_proba(trainX)
    rbf_pred=bagging_rbf_svm.predict_proba(trainX)
    blended_train_pred = (rbf_pred + linear_pred) / 2
        
    # Train a meta-classifier on blended predictions
    meta_classifier = LogisticRegression(random_state=27)
    meta_classifier.fit(blended_train_pred, trainy)

    # Predict on test set
    rbf_test_pred = bagging_rbf_svm.predict_proba(trainX)
    linear_test_pred = bagging_linear_svm.predict_proba(trainX)
    blended_test_pred = (rbf_test_pred + linear_test_pred) / 2

    # Final predictions using meta-classifier
    final_predictions = meta_classifier.predict(blended_test_pred)
    # Evaluate the model
    # new_linear_pred=bagging_linear_svm.predict(trainX)
    # new_rbf_pred=bagging_rbf_svm.predict(trainX)

    trainy_array=trainy.values
    # print(testy_array.shape)
    accuracy = accuracy_score(trainy_array, final_predictions)
    report = classification_report(trainy_array, final_predictions)

    print(
        'rbf_bagging_svm:', bagging_rbf_svm,
        'linear_bagging_svm:', bagging_linear_svm,
        'meta_classifier:', meta_classifier,
        'accuracy:', accuracy,
        'classification_report:', report
    )
    rbf_test_pred = bagging_rbf_svm.predict_proba(testX)
    linear_test_pred = bagging_linear_svm.predict_proba(testX)
    blended_test_pred = (rbf_test_pred + linear_test_pred) / 2

    # Final predictions using meta-classifier
    final_predictions = meta_classifier.predict(blended_test_pred)

    testy_array=testy.values
    accuracy = accuracy_score(testy_array, final_predictions)
    report = classification_report(testy_array, final_predictions)

    print(
        'rbf_bagging_svm:', bagging_rbf_svm,
        'linear_bagging_svm:', bagging_linear_svm,
        'meta_classifier:', meta_classifier,
        'accuracy:', accuracy,
        'classification_report:', report
    )
    
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: thebest(trial, trainX, trainy, testX, testy), n_trials=50)
print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)