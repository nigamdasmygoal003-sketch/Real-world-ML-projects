import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,root_mean_squared_error


data = pd.read_csv("medical.csv")
data["sex"] = data["sex"].map({"female":0,"male":1})
data["smoker"] = data["smoker"].map({"yes":1,"no":0})

x = data.drop(columns=["charges"])
y = data["charges"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

numarical_cols = x_train.select_dtypes(include=["int64","float64"]).columns
catgorical_cols = x_train.select_dtypes(include="object").columns

scaler = StandardScaler()
x_train_num = scaler.fit_transform(x_train[numarical_cols])
x_test_num = scaler.transform(x_test[numarical_cols])

encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
x_train_cat = encoder.fit_transform(x_train[catgorical_cols])
x_test_cat = encoder.transform(x_test[catgorical_cols])

x_train_final = np.hstack([x_train_num,x_train_cat])
x_test_final = np.hstack([x_test_num,x_test_cat])

model = RandomForestRegressor(n_estimators=200,max_depth=15,random_state=42,n_jobs=-1)

model.fit(x_train_final,y_train)

y_pred = model.predict(x_test_final)

r2 = r2_score(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

print(f"Random Foreste Regressor R2-Score:{r2} and RMSE:{rmse}")

joblib.dump(scaler,"scaler.pkl")
joblib.dump(encoder,"encoder.pkl")
joblib.dump(model,"model.pkl")