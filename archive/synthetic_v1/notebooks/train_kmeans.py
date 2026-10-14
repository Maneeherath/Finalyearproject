# ============================================================
#  K-MEANS CLUSTERING FOR GAMER ARCHETYPES
#  Dataset: Gaming Behavior and Cognitive Performance Dataset
#  Model:   scikit-learn KMeans
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/gaming_dataset_v2.csv')
MODEL_DIR = os.path.join(BASE_DIR, '../models')

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("K-MEANS CLUSTERING FOR GAMER ARCHETYPES")
print("=" * 60)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} student records.")

# Features for clustering
cluster_features = [
    'total_hours_week',
    'sleep_hours',
    'gpa',
    'attention_score',
    'memory_score',
    'reaction_time'
]

X = df[cluster_features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Add cluster labels to original df
df['cluster'] = kmeans.labels_

# Analyze cluster characteristics to identify archetypes
cluster_stats = df.groupby('cluster')[cluster_features].mean()
print("\nCluster Means:")
print(cluster_stats)

# Profiling each cluster
for i in range(4):
    c_df = df[df['cluster'] == i]
    hours = c_df['total_hours_week'].mean()
    sleep = c_df['sleep_hours'].mean()
    gpa = c_df['gpa'].mean()
    att = c_df['attention_score'].mean()
    mem = c_df['memory_score'].mean()
    rt = c_df['reaction_time'].mean()
    
    print(f"\nCluster {i} Profiling:")
    print(f"  Gaming Hours/Wk : {hours:.2f}")
    print(f"  Sleep Hours/Night: {sleep:.2f}")
    print(f"  Current GPA     : {gpa:.2f}")
    print(f"  Attention Score  : {att:.2f}")
    print(f"  Memory Score     : {mem:.2f}")
    print(f"  Reaction Time    : {rt:.2f} ms")

# Save models
joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'kmeans_scaler.pkl'))
joblib.dump(cluster_features, os.path.join(MODEL_DIR, 'kmeans_features.pkl'))

print("\n[SUCCESS] Saved KMeans models to models/")
print("=" * 60)
