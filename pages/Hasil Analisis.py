import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv("semarang_resto_dataset.csv")

# Cek nilai null
if df.isnull().sum().sum() > 0:
    print("⚠️ Terdapat missing value. Mengisi dengan modus atau median...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Buat label target: 1 jika rating ≥ 4.5, else 0
df["high_rating"] = (df["resto_rating"] >= 4.5).astype(int)

# Drop kolom tidak relevan
drop_cols = ["resto_id", "resto_name", "resto_rating", "resto_address"]
df = df.drop(columns=drop_cols)

# Encode semua kolom kategorikal
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Pisahkan fitur dan target
X = df.drop(columns=["high_rating"])
y = df["high_rating"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Inisialisasi dan latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Simpan model & data uji
joblib.dump(model, "rating_classifier.pkl")
X_test.to_csv("X_test_rating.csv", index=False)
y_test.to_csv("y_test_rating.csv", index=False)

print("✅ Model dan data uji berhasil disimpan.")
