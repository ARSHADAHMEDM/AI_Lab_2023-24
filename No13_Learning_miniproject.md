# Ex.No: 13 Loan Approval Prediction – Use Supervised Learning  
### DATE:                                                             
### REGISTER NUMBER : 212222040014

## AIM

The aim of this project is to develop a machine learning-based heart disease prediction system. The system uses a dataset of patient health metrics to predict the likelihood of heart disease, assisting healthcare professionals in making informed decisions based on the analysis of key risk factors.

## Algorithm

1. Start the program.

2. Import necessary packages
Import libraries, including:

3. NumPy for numerical operations,
Pandas for data handling,
Sklearn (specifically modules for model building, evaluation, and preprocessing),
Streamlit for creating the user interface.
Load the loan dataset
Use pd.read_csv() in Pandas to load the loan dataset into a DataFrame.

4. Check for missing values in the dataset
Identify missing values using df.isnull().sum(). Handle missing values either by filling with median values (using fillna(df.median())) or dropping rows if necessary.

5. Encode categorical features
If the dataset contains categorical variables, convert them into numerical values using techniques like one-hot encoding or label encoding with Sklearn’s LabelEncoder or pd.get_dummies().

6. Split the data into input features (X) and target label (Y)
Define input features X (the predictor variables) and target label Y (e.g., Loan_Status, indicating loan approval).

7. Divide the data into training and testing sets
Use Sklearn’s train_test_split to split the data into training and testing sets (e.g., 80% training and 20% testing).

8. Standardize the features
Use Sklearn’s StandardScaler to standardize the features in both the training and testing sets to have a mean of 0 and standard deviation of 1.

9. Choose and train a classifier
Select a classifier such as Logistic Regression or Random Forest. Train the model on the training data using model.fit(X_train, Y_train).

10. Evaluate model accuracy
Make predictions on the test set using model.predict(X_test). Calculate the accuracy using accuracy_score from Sklearn to evaluate how well the model performs.

11. Deploy the Streamlit app for online access
Deploy the application, making it accessible to users who want to interact with the loan approval prediction model.

12. Stop the program.

## Program

```python
import pandas as pd 
import plotly.express as px
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/kaggle/input/loan-default/Loan_default.csv')
df

df.info()

df.isnull().sum()

eda = ProfileReport(df , dark_mode= True , title = 'EDA REPORT ')
eda

# DROPING THE ID COLUMN SINCE IT IS NOT USEFUL FOR MODELING 
df.drop('LoanID' ,axis = 1 , inplace=True)

df.columns

col = ['Income', 'LoanAmount']

for i in col:
    fig = px.box(df , y = i , title  = f'BOX PLOT FOR {i}')
    fig.show()

# FEATURES ENGINEERING 

df['LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']

df['CreditUtilizationRate'] = df['LoanAmount'] / df['CreditScore']

col = ['LoanToIncomeRatio' ,'CreditUtilizationRate']

for i in col:
    fig = px.box(df , y=i)
    fig.show()

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# List of binary columns
binary_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']

# Apply LabelEncoder to binary columns
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# One-hot encoding categorical columns
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']

# Apply one-hot encoding
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True).astype(float)  # drop_first=True avoids multicollinearity

from sklearn.preprocessing import MinMaxScaler

# List of numerical columns to scale
numerical_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                     'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'LoanToIncomeRatio' , 'CreditUtilizationRate']

# Initialize scaler
scaler = MinMaxScaler()


# Fit and transform the numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Check the scaled data
df.head()

# Import necessary library
from sklearn.model_selection import train_test_split

# Split the dataset into features and target variable
X = df.drop('Default', axis=1)  # Features
y = df['Default']  # Target variable

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the resulting datasets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Import necessary libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
dt_predictions = dt_model.predict(X_test)

# Evaluation
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Classification Report:\n", classification_report(y_test, dt_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluation
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
gb_predictions = gb_model.predict(X_test)

# Evaluation
print("Gradient Boosting Classifier:")
print("Accuracy:", accuracy_score(y_test, gb_predictions))
print("Classification Report:\n", classification_report(y_test, gb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, gb_predictions))

# XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test set
xgb_predictions = xgb_model.predict(X_test)

# Evaluation
print("XGBoost Classifier:")
print("Accuracy:", accuracy_score(y_test, xgb_predictions))
print("Classification Report:\n", classification_report(y_test, xgb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_predictions))

# Extra Trees Classifier
et_model = ExtraTreesClassifier(random_state=42)
et_model.fit(X_train, y_train)

# Predict on the test set
et_predictions = et_model.predict(X_test)

# Evaluation
print("Extra Trees Classifier:")
print("Accuracy:", accuracy_score(y_test, et_predictions))
print("Classification Report:\n", classification_report(y_test, et_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, et_predictions))

# CatBoost Classifier
cat_model = CatBoostClassifier(silent=True, random_state=42)
cat_model.fit(X_train, y_train)

# Predict on the test set
cat_predictions = cat_model.predict(X_test)

# Evaluation
print("CatBoost Classifier:")
print("Accuracy:", accuracy_score(y_test, cat_predictions))
print("Classification Report:\n", classification_report(y_test, cat_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, cat_predictions))

# Performance summary
results = {
    "Model": ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "AdaBoost", "Extra Trees", "CatBoost"],
    "Accuracy": [
        accuracy_score(y_test, dt_predictions),
        accuracy_score(y_test, rf_predictions),
        accuracy_score(y_test, gb_predictions),
        accuracy_score(y_test, xgb_predictions),
        accuracy_score(y_test, ada_predictions),
        accuracy_score(y_test, et_predictions),
        accuracy_score(y_test, cat_predictions)
    ]
}


results_df = pd.DataFrame(results)
results_df.sort_values(by='Accuracy', ascending=False)



```

### Output:

![alt text](image-5.png)



### Result:
Thus the system was trained successfully and the prediction was carried out.
