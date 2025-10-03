# First, do 
# wget https://calmcode.io/static/data/titanic.csv
# to get the .csv file containing the titanic data

import pandas as pd
import numpy as np
import xgboost
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

def accuracy(true_values, predictions):
    return np.mean(true_values == predictions)

df = pd.read_csv('titanic.csv')
df = df[["pclass","sex","age","survived"]]
df.dropna(inplace=True)

features = df.columns

# Column sex is type string. Encode it to numerical
label_encoder = LabelEncoder()
df["sex"] = label_encoder.fit_transform(df["sex"])

# Check which sex was labelled as what
classes = label_encoder.classes_
label_mapping = {label: idx for idx, label in enumerate(classes)}
print(label_mapping)

# Train test split
X_data = df[['pclass', 'sex', 'age']].values
y_data = df['survived'].values
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

if False:
    # Predict and evaluate model 
    predictions = model.predict(X_test)
    print("Accuracy: ", accuracy(y_test, predictions))
    print("Precision: ", metrics.precision_score(y_test, predictions))
    print("Sensitivity / recall: ", metrics.recall_score(y_test, predictions))

    probs  = model.predict_proba(X_test)[:,1]

    plt.hist(y_test, color="green", alpha=0.5, label="targets")
    plt.hist(probs, color="blue", alpha=0.5, label="predictions")
    plt.legend()
    plt.show()

# Counterfactual
X_paul = [2, 1, 25]
pred_paul = model.predict_proba([X_paul])
print("Paul's chance for survival is: ", pred_paul[0][1])

print(model.predict_proba([[1,1,25]])[0][1])
print(model.predict_proba([[2,0,25]])[0][1])
print(model.predict_proba([[2,1,12]])[0][1])
