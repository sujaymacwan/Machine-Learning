import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer 
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

# Load data
data = pd.read_csv("great_customers.csv")

# List of columns with missing values
columns_with_missing = ['age', 'workclass', 'salary', 'occupation', 'mins_beerdrinking_year', 'mins_exercising_year', 'tea_per_year', 'coffee_per_year']

# Impute missing values
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

data[['age', 'salary', 'mins_beerdrinking_year', 'mins_exercising_year', 'tea_per_year', 'coffee_per_year']] = numerical_imputer.fit_transform(data[['age', 'salary', 'mins_beerdrinking_year', 'mins_exercising_year', 'tea_per_year', 'coffee_per_year']])
data[['workclass', 'occupation']] = categorical_imputer.fit_transform(data[['workclass', 'occupation']])

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill NaN values based on specific strategies (choose based on your data)
data.fillna(data.mean(numeric_only=True), inplace=True)  
data.fillna(data.mode().iloc[0], inplace=True)  

# Selecting independent columns
X = data.drop(columns=['user_id', 'great_customer_class'])
y = data['great_customer_class']

# Applying one-hot encoding to handle categorical variables
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_columns)],
                                 remainder='passthrough')
X_transformed = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Apply StandardScaler to scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build models
random_forest_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
naive_bayes_model = GaussianNB()
knn_model = KNeighborsClassifier()

# Train models
random_forest_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)
logistic_model.fit(X_train_scaled, y_train)  
naive_bayes_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_rf = random_forest_model.predict(X_test_scaled)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_logistic = logistic_model.predict(X_test_scaled)  
y_pred_nb = naive_bayes_model.predict(X_test_scaled)
y_pred_knn = knn_model.predict(X_test_scaled)

# Ensemble using Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('Random Forest', random_forest_model),
    ('SVM', svm_model),
    ('Logistic Regression', logistic_model),
    ('Naive Bayes', naive_bayes_model),
    ('KNN', knn_model)
], voting='hard')

ensemble_model.fit(X_train, y_train)

# Make predictions on the test set using the ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)

# Print accuracy and classification report for each model
models = {
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svm,
    'Logistic Regression': y_pred_logistic,
    'Naive Bayes': y_pred_nb,
    'KNN': y_pred_knn,
    'Ensemble (Voting)': y_pred_ensemble
}

for model_name, y_pred in models.items():
    print(f"\n{model_name} Model:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))