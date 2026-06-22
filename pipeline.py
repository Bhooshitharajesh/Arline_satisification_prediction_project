import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("data\DS-DATA.csv")

df['Arrival Delay'] = df['Arrival Delay'].fillna(df['Arrival Delay'].median())


df.drop('ID', axis=1, inplace=True)


df['Flight Distance'] = df['Flight Distance'].astype(str).str.replace(r'[^0-9]', '', regex=True)


df['Flight Distance'] = pd.to_numeric(df['Flight Distance'])

df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0,25,40,60,100],
    labels=['Young','Adult','Middle_Aged','Senior']
)

df['Delay_Diff'] = df['Arrival Delay'] - df['Departure Delay']

service_cols = [
    'Departure and Arrival Time Convenience',
    'Ease of Online Booking',
    'Check-in Service',
    'Online Boarding',
    'Gate Location',
    'On-board Service',
    'Seat Comfort',
    'Leg Room Service',
    'Cleanliness',
    'Food and Drink',
    'In-flight Service',
    'In-flight Wifi Service',
    'In-flight Entertainment',
    'Baggage Handling'
]

df['Total_Service_Score'] = df[service_cols].sum(axis=1)

df['Satisfaction'] = df['Satisfaction'].map({
    'Neutral or Dissatisfied': 0,
    'Satisfied': 1
})

df['Gender'] = df['Gender'].map({
    'Female': 0,
    'Male': 1
})

df['Customer Type'] = df['Customer Type'].map({
    'First-time': 0,
    'Returning': 1
})

df['Type of Travel'] = df['Type of Travel'].map({
    'Personal': 0,
    'Business': 1
})

df = pd.get_dummies(
    df,
    columns=['Class', 'Age_Group'],
    drop_first=True
)

bool_cols = df.select_dtypes(include='bool').columns

df[bool_cols] = df[bool_cols].astype(int)

X = df.drop('Satisfaction', axis=1)
y = df['Satisfaction']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

import joblib

joblib.dump(rf, 'airline_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
joblib.dump(service_cols, 'service_cols.pkl')

print("model loaded and saved successfully")




