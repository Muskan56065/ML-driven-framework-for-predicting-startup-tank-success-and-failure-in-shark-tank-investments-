import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

df = pd.read_csv("dataset/startup_success.csv")

df['status'] = df['status'].map({'acquired':1, 'closed':0})

print("Before balance:")
print(df['status'].value_counts())

major = df[df.status == 1]
minor = df[df.status == 0]

minor_up = resample(minor,
                    replace=True,
                    n_samples=len(major),
                    random_state=42)

df_balanced = pd.concat([major, minor_up])

X = df_balanced.drop('status', axis=1)
y = df_balanced['status']

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "FINAL_MODEL.pkl")
print("FINAL MODEL CREATED")
