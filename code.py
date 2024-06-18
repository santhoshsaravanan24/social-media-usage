# Import necessary libraries
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

data  = pd.read_csv("clean_social.csv").dropna()

X = data.drop(["id","socialDuration"],axis=1)
y = data["socialDuration"]
categories = ["gender",	"education"	,"profession"	,"workDuration","typeSocial","useSocial","productivity"]


c = ColumnTransformer([
    ("encode",OneHotEncoder(),categories)
    ])

model = Pipeline([
    ("columns",c),
    ("model",Ridge(alpha=0.001,max_iter=1000))
])

model.fit(X,y)

y_pred = model.predict(X)
print("R2:", r2_score(y,y_pred))
print("MSE:",mean_squared_error(y,y_pred))