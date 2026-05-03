import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

#Load Dataset

data = fetch_california_housing()
x = pd.DataFrame(data.data,columns=data.feature_names)
y = data.target

#split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#scaled

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)

#model

model = Ridge(alpha=1.0)
model.fit(x_train_scaler,y_train)

#saved model

joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")