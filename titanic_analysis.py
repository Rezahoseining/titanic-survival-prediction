# ==========================================
# TITANIC SURVIVAL PREDICTION
# A beginner-friendly Data Science project
# Author: [Reza Hosseini Tazangi]
# ==========================================

# 1. Load and explore data
# 2. Clean and prepare data
# 3. Train a Decision Tree model
# 4. Evaluate and interpret results


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# use DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load and explore data
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# 2. Clean and prepare data
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
df = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# 3. Train a Decision Tree model
y = df['Survived']
X = df.drop('Survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"model accuracy: {accuracy:.2f} means {accuracy*100:.0f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 4. Evaluate and interpret results
feature_importances = pd.DataFrame({'feature': X.columns,'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print("\nThe most important features:")
print(feature_importances)