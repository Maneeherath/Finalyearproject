"""
ANN (Artificial Neural Network) for Cognitive Performance Prediction
Replaces the Ridge Regression baseline reported in Chapter 3 of the thesis.

Two outputs: attention_score, memory_score
Trained + evaluated with the SAME 5-fold CV protocol used for the Random Forest
classifier, so results are directly comparable in the progress review.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json

tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv("data/gaming_dataset_v2.csv")

# ---- Encoding (same scheme as thesis Section 1.8.2) ----
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
df['degree_program'] = df['degree_program'].map(
    {'Business': 0, 'CS': 1, 'Engineering': 2, 'IT': 3, 'Medicine': 4})
game_dummies = pd.get_dummies(df['game_type'], prefix='game')
df = pd.concat([df, game_dummies], axis=1)

feature_cols = ['age', 'gender', 'year_of_study', 'degree_program',
                 'years_gaming', 'sessions_per_week', 'session_hours',
                 'total_hours_week', 'night_playing', 'sleep_hours',
                 'social_gaming', 'gpa'] + list(game_dummies.columns)

X_raw = df[feature_cols].astype(float).values
y_att = df['attention_score'].values
y_mem = df['memory_score'].values

scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X_raw)

scaler_att = MinMaxScaler()
scaler_mem = MinMaxScaler()
y_att_s = scaler_att.fit_transform(y_att.reshape(-1, 1)).ravel()
y_mem_s = scaler_mem.fit_transform(y_mem.reshape(-1, 1)).ravel()

def build_ann(n_features):
    # Small, heavily-regularised network: with only ~300 samples a deep
    # 64-32-16 net overfits badly (verified empirically). A shallow net
    # with L2 + dropout generalises far better at this dataset size.
    reg = tf.keras.regularizers.l2(0.01)
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(16, activation='relu', kernel_regularizer=reg)(inp)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(8, activation='relu', kernel_regularizer=reg)(x)
    att_out = layers.Dense(1, activation='linear', name='attention')(x)
    mem_out = layers.Dense(1, activation='linear', name='memory')(x)
    model = models.Model(inputs=inp, outputs=[att_out, mem_out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss={'attention': 'mse', 'memory': 'mse'},
                  loss_weights={'attention': 1.0, 'memory': 1.0})
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)

ann_att_r2, ann_mem_r2, ann_att_rmse, ann_mem_rmse = [], [], [], []
ridge_att_r2, ridge_mem_r2, ridge_att_rmse, ridge_mem_rmse = [], [], [], []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
    X_tr, X_te = X[tr_idx], X[te_idx]
    yatt_tr, yatt_te = y_att_s[tr_idx], y_att_s[te_idx]
    ymem_tr, ymem_te = y_mem_s[tr_idx], y_mem_s[te_idx]

    # --- ANN ---
    model = build_ann(X.shape[1])
    es = callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit(X_tr, {'attention': yatt_tr, 'memory': ymem_tr},
              epochs=200, batch_size=16, verbose=0, callbacks=[es])
    pred_att, pred_mem = model.predict(X_te, verbose=0)

    # inverse-transform to original score scale for interpretable RMSE
    pred_att_orig = scaler_att.inverse_transform(pred_att).ravel()
    pred_mem_orig = scaler_mem.inverse_transform(pred_mem).ravel()
    true_att_orig = scaler_att.inverse_transform(yatt_te.reshape(-1, 1)).ravel()
    true_mem_orig = scaler_mem.inverse_transform(ymem_te.reshape(-1, 1)).ravel()

    ann_att_r2.append(r2_score(true_att_orig, pred_att_orig))
    ann_mem_r2.append(r2_score(true_mem_orig, pred_mem_orig))
    ann_att_rmse.append(np.sqrt(mean_squared_error(true_att_orig, pred_att_orig)))
    ann_mem_rmse.append(np.sqrt(mean_squared_error(true_mem_orig, pred_mem_orig)))

    # --- Ridge baseline (thesis's original model, same fold) ---
    r_att = Ridge(alpha=1.0).fit(X_tr, yatt_tr)
    r_mem = Ridge(alpha=1.0).fit(X_tr, ymem_tr)
    p_att = scaler_att.inverse_transform(r_att.predict(X_te).reshape(-1, 1)).ravel()
    p_mem = scaler_mem.inverse_transform(r_mem.predict(X_te).reshape(-1, 1)).ravel()
    ridge_att_r2.append(r2_score(true_att_orig, p_att))
    ridge_mem_r2.append(r2_score(true_mem_orig, p_mem))
    ridge_att_rmse.append(np.sqrt(mean_squared_error(true_att_orig, p_att)))
    ridge_mem_rmse.append(np.sqrt(mean_squared_error(true_mem_orig, p_mem)))

    print(f"Fold {fold}: ANN attention R2={ann_att_r2[-1]:.4f}  "
          f"ANN memory R2={ann_mem_r2[-1]:.4f}  "
          f"Ridge attention R2={ridge_att_r2[-1]:.4f}  Ridge memory R2={ridge_mem_r2[-1]:.4f}")

results = {
    "ANN": {
        "attention_R2_mean": float(np.mean(ann_att_r2)), "attention_R2_std": float(np.std(ann_att_r2)),
        "memory_R2_mean": float(np.mean(ann_mem_r2)), "memory_R2_std": float(np.std(ann_mem_r2)),
        "attention_RMSE_mean": float(np.mean(ann_att_rmse)),
        "memory_RMSE_mean": float(np.mean(ann_mem_rmse)),
    },
    "Ridge": {
        "attention_R2_mean": float(np.mean(ridge_att_r2)), "attention_R2_std": float(np.std(ridge_att_r2)),
        "memory_R2_mean": float(np.mean(ridge_mem_r2)), "memory_R2_std": float(np.std(ridge_mem_r2)),
        "attention_RMSE_mean": float(np.mean(ridge_att_rmse)),
        "memory_RMSE_mean": float(np.mean(ridge_mem_rmse)),
    }
}

print("\n===== 5-FOLD CV SUMMARY (comparable to thesis Table, Section 2.5.2) =====")
print(json.dumps(results, indent=2))

with open('ann_vs_ridge_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Train final ANN on all data for deployment (used by Streamlit app)
final_model = build_ann(X.shape[1])
final_model.fit(X, {'attention': y_att_s, 'memory': y_mem_s}, epochs=200, batch_size=16, verbose=0,
                callbacks=[callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)])
final_model.save('models/cognitive_ann_model.keras')
print("\nSaved final deployment model -> cognitive_ann_model.keras")