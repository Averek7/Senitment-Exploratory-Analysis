from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import files

drive.mount('/content/drive')

train = pd.read_csv("train.csv")

train.head()

# Display Data Sets


train.isnull()

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')

sns.displot(train['Age'].dropna(), kde=False, color='darkred', bins=40)

train['Age'].hist(bins=30, color='darkred', alpha=0.3)

sns.countplot(x='SibSp', data=train)

train['Fare'].hist(color='green', bins=40, figsize=(8, 4))

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')


def input_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(input_age, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

train.drop('Cabin', axis=1, inplace=True)

train.head()

train.dropna(inplace=True)

train.info()

pd.get_dummies(train['Embarked'], drop_first=True).head()

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

train.head()

train = pd.concat([train, sex, embark], axis=1)

train.head()

# Building Logistic Regression Model

train.drop('Survived', axis=1).head()

train['Survived'].head()


X_train, X_test, y_train, y_test = train_test_split(train.drop(
    'Survived', axis=1), train['Survived'], test_size=0.30, random_state=101)

# Training & Predicting


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)


accuracy = confusion_matrix(y_test, predictions)

accuracy


accuracy = accuracy_score(y_test, predictions)
accuracy

predictions
