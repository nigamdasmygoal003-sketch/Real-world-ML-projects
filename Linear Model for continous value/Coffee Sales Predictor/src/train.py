import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,root_mean_squared_error


def load_data(path):
    data = pd.read_csv(path)
    x = data.drop(columns=["cash_type","money","Date","Time"])
    y = data["money"]
    
    return x,y

def load_model(x):
    num_features = x.select_dtypes(include=["int64","float64"]).columns
    cat_features = x.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="mean")),
        ("scaler",StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num",num_pipeline,num_features),
        ("cat",cat_pipeline,cat_features)
    ])
    
    model = RandomForestRegressor(n_estimators=200,
                                  max_depth=50,
                                  random_state=42,
                                  n_jobs=-1)
    
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("model",model)
    ])
    
    return pipe


def train():
    print("Loading Data ...")
    path = "data/Coffe_sales.csv"
    x,y = load_data(path)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    print("\nLoad Model ...")
    model = load_model(x)
    print("\nTraining Model ...")
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(f"\nr2-score:{r2_score(y_test,pred)}")
    print(f"\nRMSE:{root_mean_squared_error(y_test,pred)}")
    joblib.dump(model,"model/model.pkl")
    print("\nTraining complete ! saved at model/model.pkl ")
    
if __name__ == "__main__":
    train()    