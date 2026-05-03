import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# STEP 1: LOAD DATA
# =========================
print("🔹 Loading dataset...")
data = pd.read_csv("car_price_prediction.csv")

# =========================
# STEP 2: CLEAN DATA
# =========================
print("🔹 Cleaning data...")

# Drop useless column
data = data.drop(columns=["ID"])

# Fix Levy (remove '-' and convert)
data["Levy"] = data["Levy"].replace("-", np.nan)
data["Levy"] = pd.to_numeric(data["Levy"], errors="coerce")

# Fix Engine volume (remove 'Turbo')
data["Engine volume"] = data["Engine volume"].str.replace("Turbo", "", regex=False)
data["Engine volume"] = pd.to_numeric(data["Engine volume"], errors="coerce")

# Fix Mileage (remove 'km')
data["Mileage"] = data["Mileage"].str.replace(" km", "", regex=False)
data["Mileage"] = pd.to_numeric(data["Mileage"], errors="coerce")

# Fix Doors (extract number)
data["Doors"] = data["Doors"].str.extract('(\d+)')
data["Doors"] = pd.to_numeric(data["Doors"], errors="coerce")

# =========================
# STEP 3: HANDLE TARGET
# =========================
data = data.dropna(subset=["Price"])

# =========================
# STEP 4: SPLIT FEATURES
# =========================
X = data.drop(columns=["Price"])
y = data["Price"]

# =========================
# STEP 5: TRAIN TEST SPLIT
# =========================
print("🔹 Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 6: COLUMN TYPES
# =========================
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# =========================
# STEP 7: NUMERICAL
# =========================
print("🔹 Processing numerical...")

num_imputer = SimpleImputer(strategy="mean")
X_train_num = num_imputer.fit_transform(X_train[num_cols])
X_test_num = num_imputer.transform(X_test[num_cols])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# =========================
# STEP 8: CATEGORICAL
# =========================
print("🔹 Processing categorical...")

cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
X_test_cat = cat_imputer.transform(X_test[cat_cols])

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

# =========================
# STEP 9: COMBINE
# =========================
X_train_final = np.hstack([X_train_num, X_train_cat])
X_test_final = np.hstack([X_test_num, X_test_cat])

print("✅ Final shape:", X_train_final.shape)

# =========================
# STEP 10: TRAIN MODEL
# =========================
print("🔹 Training model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

model.fit(X_train_final, y_train)

# =========================
# STEP 11: EVALUATE
# =========================
print("🔹 Evaluating...")

y_pred = model.predict(X_test_final)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✅ R2 Score: {r2:.4f}")
print(f"✅ RMSE: {rmse:.2f}")

# =========================
# STEP 12: SAVE
# =========================
print("🔹 Saving model...")

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(num_imputer, "num_imputer.pkl")
joblib.dump(cat_imputer, "cat_imputer.pkl")

print("🎉 DONE!")