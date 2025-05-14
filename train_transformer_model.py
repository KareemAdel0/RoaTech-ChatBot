import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# Folder where model files will be saved
pkl_folder = "model_pkl_files"

# 1. Load and preprocess data
df = pd.read_csv("datasets/track_training_cleaned_data.csv")
df["text"] = df["text"].str.lower().str.strip()

# 2. Encode labels numerically
label_encoder = LabelEncoder()
df["track_encoded"] = label_encoder.fit_transform(df["track"])

# Count how many examples exist for each track
track_counts = df["track"].value_counts()

# Filter out classes with less than 2 samples
df = df[df["track"].isin(track_counts[track_counts >= 2].index)]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["track_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["track_encoded"]
)

# 4. Load pre-trained sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Encode text into embeddings
print("Encoding training data...")
emb_train = embedder.encode(X_train.tolist(), show_progress_bar=True)
print("Encoding test data...")
emb_test = embedder.encode(X_test.tolist(), show_progress_bar=True)

# 6. Use GridSearchCV to find best C
param_grid = {"C": [0.01, 0.1, 1, 10]}
logestic_model = LogisticRegression(max_iter=1000, random_state=42)
grid_search = GridSearchCV(logestic_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(emb_train, y_train)

best_logestic_model = grid_search.best_estimator_

# 7. Evaluation
y_pred = best_logestic_model.predict(emb_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nBest C from GridSearch: {grid_search.best_params_['C']}")
print(f"Accuracy on test set: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(label_encoder.inverse_transform(y_test),
                            label_encoder.inverse_transform(y_pred)))

# 8. Save models
with open(os.path.join(pkl_folder, "embedder.pkl"), "wb") as f:
    pickle.dump(embedder, f)

with open(os.path.join(pkl_folder, "transformer_logestic_model.pkl"), "wb") as f:
    pickle.dump(best_logestic_model, f)

with open(os.path.join(pkl_folder, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\nSaved model files in: {pkl_folder}")