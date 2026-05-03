import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("StudentPerformance.csv")

x = data.drop("Performance Index",axis=1)
y = data["Performance Index"]

x["Extracurricular Activities"] = x["Extracurricular Activities"].map({"Yes":1,"No":0})

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

model = LinearRegression()
model.fit(x_train_scaled,y_train)

joblib.dump(scaler,"scaler.pkl")
joblib.dump(model,"model.pkl")