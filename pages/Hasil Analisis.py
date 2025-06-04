import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("semarang_resto_dataset.csv")

# Buat kolom target: 1 jika rating >= 4.5, else 0
df["high_rating"] = (df["resto_rating"] >= 4.5).astype(int)

# Hapus kolom yang tidak relevan
drop_cols = ["resto_id", "resto_name", "resto_rating", "resto_address"]
df.drop(columns=drop_cols, inplace=True)

# Cek dan isi missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Encode kolom kategorikal
encoders = {}
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # simpan encoder untuk digunakan di Streamlit

# Split data
X = df.drop(columns=["high_rating"])
y = df["high_rating"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, "rating_classifier.pkl")

# Simpan encoder
os.makedirs("encoders", exist_ok=True)
for col, encoder in encoders.items():
    joblib.dump(encoder, f"encoders/{col}_encoder.pkl")

# Simpan data uji
X_test.to_csv("X_test_rating.csv", index=False)
y_test.to_csv("y_test_rating.csv", index=False)

print("✅ Model, encoder, dan data uji berhasil disimpan.")
