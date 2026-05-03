import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,root_mean_squared_error


data = pd.read_csv("WineQuality.csv")
x = data.drop(columns=["quality"])
y = data["quality"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

x_train_final = scaler.fit_transform(x_train)
x_test_final = scaler.transform(x_test)

model = RandomForestRegressor(n_estimators=200,max_depth=15,random_state=42,n_jobs=-1)

model.fit(x_train_final,y_train)

y_pred = model.predict(x_test_final)

r2 = r2_score(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

print(f"Random Foreste Regressor R2-Score:{r2} and RMSE:{rmse}")

joblib.dump(scaler,"scaler.pkl")
joblib.dump(model,"model.pkl")
