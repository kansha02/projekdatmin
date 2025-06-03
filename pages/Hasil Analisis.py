import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("semarang_resto_dataset.csv")

# Buat label target: 1 jika rating ≥ 4.5, else 0
df["high_rating"] = (df["resto_rating"] >= 4.5).astype(int)

# Drop kolom tidak relevan
drop_cols = ["resto_id", "resto_name", "resto_rating", "resto_address"]
df = df.drop(columns=drop_cols)

# Encode kolom kategorikal
df["resto_type"] = LabelEncoder().fit_transform(df["resto_type"])

# Pisahkan fitur dan target
X = df.drop(columns=["high_rating"])
y = df["high_rating"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan model & data uji
joblib.dump(model, "rating_classifier.pkl")
X_test.to_csv("X_test_rating.csv", index=False)
y_test.to_csv("y_test_rating.csv", index=False)

print("✅ Model dan data uji berhasil disimpan.")
