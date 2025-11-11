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
# ما از کلاسیفیکشن استفاده میکنیم
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load and explore data
# ۱. داده رو بخوان
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# 2. Clean and prepare data
# ۲. پاک‌سازی داده (همان کارهای روز ۲ — بدون هشدار)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
df = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
# ادامه 2 . تبدیل متغیرهای متنی به عدد
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# 3. Train a Decision Tree model
# 3. آموزش برنامه
# ستون 'Survived' هدف ماست a داده های هدف را از داده ها ورودی جدا میکنیم
y = df['Survived']
X = df.drop('Survived', axis=1)
# جدا سازی داده های تست و اموزش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# مدل رو بساز
model = DecisionTreeClassifier(random_state=42)
# مدل رو آموزش بده
model.fit(X_train, y_train)
# پیش‌بینی روی داده‌ی تست
y_pred = model.predict(X_test)
# دقت مدل چقدره؟
accuracy = accuracy_score(y_test, y_pred)
print(f"دقت مدل: {accuracy:.2f} یعنی {accuracy*100:.0f}%")
# گزارش کامل‌تر
print("\nگزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))

# 4. Evaluate and interpret results
# 4. بررسی خروجی ها و درست عمل کردن برنامه
# مهم‌ترین ویژگی‌ها رو ببین
feature_importances = pd.DataFrame({'feature': X.columns,'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print("\nمهم‌ترین ویژگی‌ها:")
print(feature_importances)