import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# Load data
inputs = pd.read_csv("delhi_traffic_features.csv")
target = pd.read_csv("delhi_traffic_target.csv")

inputs = inputs.drop("Trip_ID", axis=1)
target = target.drop("Trip_ID", axis=1).squeeze()

# Split FIRST (IMPORTANT)
train_input, test_input, train_target, test_target = train_test_split(
    inputs, target, test_size=0.3, random_state=42
)

# Columns
numerical_cols = train_input.select_dtypes(include=np.number).columns
categorical_cols = train_input.select_dtypes(include=["object", "category"]).columns

# Scale
scaler = MinMaxScaler()
train_input[numerical_cols] = scaler.fit_transform(train_input[numerical_cols])
test_input[numerical_cols] = scaler.transform(test_input[numerical_cols])

# Encode
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_input[categorical_cols])

encoded_train = encoder.transform(train_input[categorical_cols])
encoded_test = encoder.transform(test_input[categorical_cols])

encoder_cols = encoder.get_feature_names_out(categorical_cols)

train_input[encoder_cols] = pd.DataFrame(encoded_train, index=train_input.index)
test_input[encoder_cols] = pd.DataFrame(encoded_test, index=test_input.index)

train_input.drop(columns=categorical_cols, inplace=True)
test_input.drop(columns=categorical_cols, inplace=True)

# Model
model = RandomForestRegressor(n_estimators=160, max_depth=50, random_state=42, n_jobs=-1)
model.fit(train_input, train_target)

# Predict
target_pred = model.predict(test_input)

# Metrics
r2 = r2_score(test_target, target_pred)
rmse = root_mean_squared_error(test_target, target_pred)

print(f"R2: {r2}, RMSE: {rmse}")

# Save
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(model, "model.pkl")


