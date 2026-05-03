# src/train.py

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def load_data(path):
    data = pd.read_csv(path)
    data.dropna(subset=["RainToday","RainTomorrow"],inplace=True)
    
    data["RainToday"] = data["RainToday"].map({"Yes":1,"No":0})
    data["RainTomorrow"] = data["RainTomorrow"].map({"Yes":1,"No":0})
    
    x = data.drop(columns=["Date","RainTomorrow"])
    y = data["RainTomorrow"]
    
    return x,y

def load_model(x):
    num_features = x.select_dtypes(include=["float64"]).columns
    cat_features = x.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])
    
    model = SVC(class_weight="balanced")
    
    pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    return pipe

def train():
    path = "data/weatherAUS.csv"
    print("Loading Data...")
    x,y = load_data(path)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    print("\nLoading Model...")
    model = load_model(x)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    class_report = classification_report(y_test,pred)
    print(f"\nClassification Report: {class_report}")
    joblib.dump(model,"model/model.pkl")
    print("\nModel saved in model/model.pkl")
    

if __name__=="__main__":
    train()    
    
    
    