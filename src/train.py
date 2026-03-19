import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/processed/final_churn.csv")

X = df.drop("Churn",axis=1)
y = df["Churn"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42)

pipeline = Pipeline([
("scaler",StandardScaler()),
("model",LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train,y_train)

joblib.dump(pipeline,"models/model_pipeline.pkl")