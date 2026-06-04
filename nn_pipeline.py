"""
Traffic Demand Prediction - Neural Network Pipeline
====================================================
Architecture: Entity Embedding + Residual MLP
Framework:    PyTorch (CPU)
Strategy:     Train on Day 48, validate on Day 49 early morning,
              multi-seed averaging, then blend with GBDT predictions.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import warnings
import time

warnings.filterwarnings('ignore')

print("=" * 70)
print("  Traffic Demand Prediction - Neural Network Pipeline")
print("=" * 70)

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
print("\n[1/8] Loading datasets...")
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")

# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
print("\n[2/8] Engineering features...")

df_all = pd.concat([train, test], ignore_index=True)

# Geohash decoding
def decode_geohash(geohash):
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for char in geohash:
        val = base32.index(char)
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                lon_interval = (mid, lon_interval[1]) if bit else (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                lat_interval = (mid, lat_interval[1]) if bit else (lat_interval[0], mid)
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2

coords = [decode_geohash(g) for g in df_all['geohash']]
df_all['lat'] = [c[0] for c in coords]
df_all['lon'] = [c[1] for c in coords]

# Geohash hierarchy
df_all['geo_p4'] = df_all['geohash'].str[:4]
df_all['geo_p5'] = df_all['geohash'].str[:5]

# Time features
df_all['hour']   = df_all['timestamp'].apply(lambda x: int(x.split(':')[0]))
df_all['minute'] = df_all['timestamp'].apply(lambda x: int(x.split(':')[1]))
df_all['time_minutes'] = df_all['hour'] * 60 + df_all['minute']
df_all['sin_time']     = np.sin(2 * np.pi * df_all['time_minutes'] / 1440)
df_all['cos_time']     = np.cos(2 * np.pi * df_all['time_minutes'] / 1440)
df_all['sin_time_12h'] = np.sin(2 * np.pi * df_all['time_minutes'] / 720)
df_all['cos_time_12h'] = np.cos(2 * np.pi * df_all['time_minutes'] / 720)

# ---------------------------------------------------------------------
# 3. IMPUTATION
# ---------------------------------------------------------------------
print("\n[3/8] Imputing missing values...")

# RoadType
road_type_map = train.groupby('geohash')['RoadType'].apply(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_road_type = train['RoadType'].mode()[0]
df_all['RoadType'] = df_all['RoadType'].fillna(df_all['geohash'].map(road_type_map)).fillna(global_road_type)

# NumberofLanes
lanes_map = train.groupby('geohash')['NumberofLanes'].apply(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lanes = train['NumberofLanes'].mode()[0]
df_all['NumberofLanes'] = df_all['NumberofLanes'].fillna(df_all['geohash'].map(lanes_map)).fillna(global_lanes)

# LargeVehicles
lv_map = train.groupby('geohash')['LargeVehicles'].apply(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lv = train['LargeVehicles'].mode()[0]
df_all['LargeVehicles'] = df_all['LargeVehicles'].fillna(df_all['geohash'].map(lv_map)).fillna(global_lv)

# Landmarks
lm_map = train.groupby('geohash')['Landmarks'].apply(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lm = train['Landmarks'].mode()[0]
df_all['Landmarks'] = df_all['Landmarks'].fillna(df_all['geohash'].map(lm_map)).fillna(global_lm)

# Temperature
geo_temp = train.groupby('geohash')['Temperature'].mean().to_dict()
ts_temp  = train.groupby('timestamp')['Temperature'].mean().to_dict()
global_temp = train['Temperature'].mean()
df_all['Temperature'] = (df_all['Temperature']
    .fillna(df_all['geohash'].map(geo_temp))
    .fillna(df_all['timestamp'].map(ts_temp))
    .fillna(global_temp))

# Weather
ts_weather = train.groupby('timestamp')['Weather'].apply(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_weather = train['Weather'].mode()[0]
df_all['Weather'] = df_all['Weather'].fillna(df_all['timestamp'].map(ts_weather)).fillna(global_weather)

print(f"  Remaining NAs: {df_all.isnull().sum().sum()}")

# ---------------------------------------------------------------------
# 4. LABEL ENCODING + DAY 48 FEATURES
# ---------------------------------------------------------------------
print("\n[4/8] Encoding categoricals and building Day 48 features...")

cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geo_p4', 'geo_p5', 'geohash']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))
    label_encoders[col] = le

# Split
train_df = df_all[df_all['demand'].notnull()].copy()
test_df  = df_all[df_all['demand'].isnull()].copy()
df_48    = train_df[train_df['day'] == 48].copy()
df_49    = train_df[train_df['day'] == 49].copy()

# Day 48 lookup tables
day48_lookup   = df_48.set_index(['geohash', 'time_minutes'])['demand'].to_dict()
geo_mean_48    = df_48.groupby('geohash')['demand'].mean().to_dict()
geo_std_48     = df_48.groupby('geohash')['demand'].std().fillna(0).to_dict()
geo_max_48     = df_48.groupby('geohash')['demand'].max().to_dict()
geo_med_48     = df_48.groupby('geohash')['demand'].median().to_dict()
ts_mean_48     = df_48.groupby('time_minutes')['demand'].mean().to_dict()
ts_std_48      = df_48.groupby('time_minutes')['demand'].std().fillna(0).to_dict()
global_mean_48 = df_48['demand'].mean()

geo_hour_mean_48 = df_48.groupby(['geohash', 'hour'])['demand'].mean().to_dict()
geo_hour_std_48  = df_48.groupby(['geohash', 'hour'])['demand'].std().fillna(0).to_dict()

p4_mean_48 = df_48.groupby('geo_p4')['demand'].mean().to_dict()
p4_std_48  = df_48.groupby('geo_p4')['demand'].std().fillna(0).to_dict()
p5_mean_48 = df_48.groupby('geo_p5')['demand'].mean().to_dict()
p5_std_48  = df_48.groupby('geo_p5')['demand'].std().fillna(0).to_dict()


def add_advanced_features(df):
    """Add all engineered features to a dataframe."""
    out = df.copy()
    
    # Day 48 lag features
    for lag_name, delta in [('prev_day_demand', 0), 
                             ('prev_day_lag15', -15), ('prev_day_lead15', 15),
                             ('prev_day_lag30', -30), ('prev_day_lead30', 30),
                             ('prev_day_lag45', -45), ('prev_day_lead45', 45),
                             ('prev_day_lag60', -60), ('prev_day_lead60', 60)]:
        out[lag_name] = out.apply(
            lambda r: day48_lookup.get((r['geohash'], r['time_minutes'] + delta), np.nan), axis=1)
    
    # Fill NaN lags
    lag_cols = ['prev_day_demand', 'prev_day_lag15', 'prev_day_lead15',
                'prev_day_lag30', 'prev_day_lead30', 'prev_day_lag45', 
                'prev_day_lead45', 'prev_day_lag60', 'prev_day_lead60']
    for c in lag_cols:
        out[c] = (out[c]
            .fillna(out['geohash'].map(geo_mean_48))
            .fillna(out['time_minutes'].map(ts_mean_48))
            .fillna(global_mean_48))
    
    # Rolling stats
    out['prev_day_rolling_mean_3'] = out[['prev_day_lag15', 'prev_day_demand', 'prev_day_lead15']].mean(axis=1)
    out['prev_day_rolling_mean_5'] = out[['prev_day_lag30', 'prev_day_lag15', 'prev_day_demand', 
                                           'prev_day_lead15', 'prev_day_lead30']].mean(axis=1)
    out['prev_day_rolling_std_3']  = out[['prev_day_lag15', 'prev_day_demand', 'prev_day_lead15']].std(axis=1)
    out['prev_day_rolling_range']  = out[['prev_day_lag30', 'prev_day_lag15', 'prev_day_demand', 
                                           'prev_day_lead15', 'prev_day_lead30']].max(axis=1) - \
                                     out[['prev_day_lag30', 'prev_day_lag15', 'prev_day_demand', 
                                           'prev_day_lead15', 'prev_day_lead30']].min(axis=1)
    
    # Geohash-level stats
    out['geo_mean_demand']   = out['geohash'].map(geo_mean_48).fillna(global_mean_48)
    out['geo_std_demand']    = out['geohash'].map(geo_std_48).fillna(0)
    out['geo_max_demand']    = out['geohash'].map(geo_max_48).fillna(global_mean_48)
    out['geo_median_demand'] = out['geohash'].map(geo_med_48).fillna(global_mean_48)
    
    # Geohash-hour stats
    out['geo_hour_mean'] = out.apply(
        lambda r: geo_hour_mean_48.get((r['geohash'], r['hour']), geo_mean_48.get(r['geohash'], global_mean_48)), axis=1)
    out['geo_hour_std']  = out.apply(
        lambda r: geo_hour_std_48.get((r['geohash'], r['hour']), 0), axis=1)
    
    # Timestamp-level stats
    out['ts_mean_demand'] = out['time_minutes'].map(ts_mean_48).fillna(global_mean_48)
    out['ts_std_demand']  = out['time_minutes'].map(ts_std_48).fillna(0)
    
    # Spatial neighborhood
    out['geo_p4_mean'] = out['geo_p4'].map(p4_mean_48).fillna(global_mean_48)
    out['geo_p4_std']  = out['geo_p4'].map(p4_std_48).fillna(0)
    out['geo_p5_mean'] = out['geo_p5'].map(p5_mean_48).fillna(global_mean_48)
    out['geo_p5_std']  = out['geo_p5'].map(p5_std_48).fillna(0)
    
    # Interactions
    out['demand_ratio_vs_geo'] = out['prev_day_demand'] / (out['geo_mean_demand'] + 1e-6)
    out['demand_diff_vs_geo']  = out['prev_day_demand'] - out['geo_mean_demand']
    out['demand_ratio_vs_ts']  = out['prev_day_demand'] / (out['ts_mean_demand'] + 1e-6)
    out['demand_diff_vs_hourly'] = out['prev_day_demand'] - out['geo_hour_mean']
    
    return out

print("  Adding features to all splits...")
train_df = add_advanced_features(train_df)
test_df  = add_advanced_features(test_df)
df_49    = add_advanced_features(df_49)

print(f"  Total features: {len(train_df.columns)}")

# ---------------------------------------------------------------------
# 5. NEURAL NETWORK DEFINITION
# ---------------------------------------------------------------------
print("\n[5/8] Defining Neural Network architecture...")

# Cardinalities for embeddings
n_geohash    = df_all['geohash'].nunique()
n_geo_p5     = df_all['geo_p5'].nunique()
n_geo_p4     = df_all['geo_p4'].nunique()
n_road_type  = df_all['RoadType'].nunique()
n_weather    = df_all['Weather'].nunique()
n_large_veh  = df_all['LargeVehicles'].nunique()
n_landmarks  = df_all['Landmarks'].nunique()

# Embedding dimensions
EMB_GEOHASH   = 32
EMB_GEO_P5    = 16
EMB_GEO_P4    = 8
EMB_ROADTYPE  = 4
EMB_WEATHER   = 4
EMB_LARGEVEH  = 2
EMB_LANDMARKS = 2

# Continuous features for the NN
continuous_features = [
    'lat', 'lon', 'time_minutes', 'sin_time', 'cos_time',
    'sin_time_12h', 'cos_time_12h', 'Temperature', 'NumberofLanes',
    # Day 48 lag features
    'prev_day_demand', 'prev_day_lag15', 'prev_day_lead15',
    'prev_day_lag30', 'prev_day_lead30', 'prev_day_lag45', 'prev_day_lead45',
    'prev_day_lag60', 'prev_day_lead60',
    # Rolling stats
    'prev_day_rolling_mean_3', 'prev_day_rolling_mean_5',
    'prev_day_rolling_std_3', 'prev_day_rolling_range',
    # Geohash stats
    'geo_mean_demand', 'geo_std_demand', 'geo_max_demand', 'geo_median_demand',
    'geo_hour_mean', 'geo_hour_std',
    # Timestamp stats
    'ts_mean_demand', 'ts_std_demand',
    # Spatial
    'geo_p4_mean', 'geo_p4_std', 'geo_p5_mean', 'geo_p5_std',
    # Interactions
    'demand_ratio_vs_geo', 'demand_diff_vs_geo', 'demand_ratio_vs_ts',
    'demand_diff_vs_hourly',
]

# Categorical feature columns (in order for the NN)
cat_feature_cols = ['geohash', 'geo_p5', 'geo_p4', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']


class ResidualBlock(nn.Module):
    """Residual block with pre-activation BatchNorm."""
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Project shortcut if dims change
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn(x)
        out = self.act(self.linear1(out))
        out = self.dropout(out)
        out = self.linear2(out)
        return self.act(out + residual)


class DemandNet(nn.Module):
    """Entity Embedding + Residual MLP for demand prediction."""
    
    def __init__(self, n_continuous):
        super().__init__()
        
        # Entity embeddings
        self.emb_geohash   = nn.Embedding(n_geohash + 1,   EMB_GEOHASH)
        self.emb_geo_p5    = nn.Embedding(n_geo_p5 + 1,    EMB_GEO_P5)
        self.emb_geo_p4    = nn.Embedding(n_geo_p4 + 1,    EMB_GEO_P4)
        self.emb_road_type = nn.Embedding(n_road_type + 1,  EMB_ROADTYPE)
        self.emb_weather   = nn.Embedding(n_weather + 1,    EMB_WEATHER)
        self.emb_large_veh = nn.Embedding(n_large_veh + 1,  EMB_LARGEVEH)
        self.emb_landmarks = nn.Embedding(n_landmarks + 1,  EMB_LANDMARKS)
        
        total_emb_dim = EMB_GEOHASH + EMB_GEO_P5 + EMB_GEO_P4 + EMB_ROADTYPE + EMB_WEATHER + EMB_LARGEVEH + EMB_LANDMARKS
        total_input_dim = total_emb_dim + n_continuous
        
        # MLP with residual blocks
        self.input_bn = nn.BatchNorm1d(total_input_dim)
        self.fc1 = nn.Linear(total_input_dim, 256)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.2)
        
        self.res1 = ResidualBlock(256, 256, dropout=0.15)
        self.res2 = ResidualBlock(256, 128, dropout=0.1)
        
        self.fc_out1 = nn.Linear(128, 64)
        self.act_out = nn.GELU()
        self.fc_out2 = nn.Linear(64, 1)
    
    def forward(self, x_cat, x_cont):
        # Embeddings
        emb = torch.cat([
            self.emb_geohash(x_cat[:, 0]),
            self.emb_geo_p5(x_cat[:, 1]),
            self.emb_geo_p4(x_cat[:, 2]),
            self.emb_road_type(x_cat[:, 3]),
            self.emb_weather(x_cat[:, 4]),
            self.emb_large_veh(x_cat[:, 5]),
            self.emb_landmarks(x_cat[:, 6]),
        ], dim=1)
        
        # Concatenate embeddings + continuous
        x = torch.cat([emb, x_cont], dim=1)
        
        # MLP
        x = self.input_bn(x)
        x = self.act1(self.fc1(x))
        x = self.drop1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        
        x = self.act_out(self.fc_out1(x))
        x = torch.sigmoid(self.fc_out2(x))  # Output in [0, 1]
        
        return x.squeeze(1)


# ---------------------------------------------------------------------
# 6. PREPARE DATA FOR NN
# ---------------------------------------------------------------------
print("\n[6/8] Preparing data tensors...")

# Normalize continuous features
scaler = StandardScaler()
X_cont_train = scaler.fit_transform(train_df[continuous_features].values.astype(np.float32))
X_cont_d49   = scaler.transform(df_49[continuous_features].values.astype(np.float32))
X_cont_test  = scaler.transform(test_df[continuous_features].values.astype(np.float32))

# Replace any NaN/inf after scaling
X_cont_train = np.nan_to_num(X_cont_train, nan=0.0, posinf=0.0, neginf=0.0)
X_cont_d49   = np.nan_to_num(X_cont_d49, nan=0.0, posinf=0.0, neginf=0.0)
X_cont_test  = np.nan_to_num(X_cont_test, nan=0.0, posinf=0.0, neginf=0.0)

# Categorical tensors
X_cat_train = train_df[cat_feature_cols].values.astype(np.int64)
X_cat_d49   = df_49[cat_feature_cols].values.astype(np.int64)
X_cat_test  = test_df[cat_feature_cols].values.astype(np.int64)

y_train = train_df['demand'].values.astype(np.float32)
y_d49   = df_49['demand'].values.astype(np.float32)

# Day 48 only (for train split in multi-seed)
mask_d48 = train_df['day'] == 48
X_cont_d48_only = X_cont_train[mask_d48.values]
X_cat_d48_only  = X_cat_train[mask_d48.values]
y_d48_only      = y_train[mask_d48.values]

print(f"  Train (Day 48): {X_cont_d48_only.shape[0]} rows")
print(f"  Val (Day 49):   {X_cont_d49.shape[0]} rows")
print(f"  Test:           {X_cont_test.shape[0]} rows")
print(f"  Continuous features: {len(continuous_features)}")
print(f"  Categorical features: {len(cat_feature_cols)}")


# ---------------------------------------------------------------------
# 7. TRAIN NN WITH MULTI-SEED
# ---------------------------------------------------------------------
print("\n[7/8] Training Neural Networks (multi-seed)...")

NUM_SEEDS = 5
BATCH_SIZE = 2048
MAX_EPOCHS = 300
PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4

nn_preds_test_all = []
nn_preds_d49_all  = []

for seed_idx, seed in enumerate([42, 123, 2024, 777, 314]):
    t0 = time.time()
    print(f"\n  -- Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed}) --")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DemandNet(n_continuous=len(continuous_features))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=0.05)  # Tight delta for skewed demand
    
    # DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_cat_d48_only),
        torch.from_numpy(X_cont_d48_only),
        torch.from_numpy(y_d48_only)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation tensors
    val_cat  = torch.from_numpy(X_cat_d49)
    val_cont = torch.from_numpy(X_cont_d49)
    val_y    = torch.from_numpy(y_d49)
    
    best_r2 = -999
    best_state = None
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch_cat, batch_cont, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_cat, batch_cont)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_cat, val_cont).numpy()
        val_r2 = r2_score(y_d49, val_preds)
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.6f}, val_R2={val_r2:.4f}, best_R2={best_r2:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}, best R2={best_r2:.4f}")
            break
    
    # Load best model and predict
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        # Predict on Day 49
        preds_d49 = model(val_cat, val_cont).numpy()
        nn_preds_d49_all.append(preds_d49)
        
        # Predict on test
        test_cat  = torch.from_numpy(X_cat_test)
        test_cont = torch.from_numpy(X_cont_test)
        preds_test = model(test_cat, test_cont).numpy()
        nn_preds_test_all.append(preds_test)
    
    elapsed = time.time() - t0
    print(f"    Final: best_R2={best_r2:.4f}, time={elapsed:.1f}s")

# Average across seeds
nn_preds_d49_avg  = np.mean(nn_preds_d49_all, axis=0)
nn_preds_test_avg = np.mean(nn_preds_test_all, axis=0)

nn_r2_d49 = r2_score(y_d49, nn_preds_d49_avg)
print(f"\n  NN Ensemble R2 on Day 49: {nn_r2_d49:.4f}")


# ---------------------------------------------------------------------
# 8. GBDT MODELS + BLEND
# ---------------------------------------------------------------------
print("\n[8/8] Training GBDT models and blending...")

features_gbdt = [
    'lat', 'lon', 'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time',
    'sin_time_12h', 'cos_time_12h',
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    'geo_p4', 'geo_p5', 'geohash',
    'prev_day_demand', 'prev_day_lag15', 'prev_day_lead15',
    'prev_day_lag30', 'prev_day_lead30', 'prev_day_lag45', 'prev_day_lead45',
    'prev_day_lag60', 'prev_day_lead60',
    'prev_day_rolling_mean_3', 'prev_day_rolling_mean_5', 
    'prev_day_rolling_std_3', 'prev_day_rolling_range',
    'geo_mean_demand', 'geo_std_demand', 'geo_max_demand', 'geo_median_demand',
    'geo_hour_mean', 'geo_hour_std',
    'ts_mean_demand', 'ts_std_demand',
    'geo_p4_mean', 'geo_p4_std', 'geo_p5_mean', 'geo_p5_std',
    'demand_ratio_vs_geo', 'demand_diff_vs_geo', 'demand_ratio_vs_ts',
    'demand_diff_vs_hourly',
]

target = 'demand'

# Model A: Train on ALL training data
print("  Training GBDT Model A (all data)...")
X_A = train_df[features_gbdt]
y_A = train_df[target]

lgb_A = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1, n_jobs=-1
)
lgb_A.fit(X_A, y_A)

xgb_A = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=7,
    reg_lambda=1.0, reg_alpha=0.1, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0, n_jobs=-1
)
xgb_A.fit(X_A, y_A)

cat_A = CatBoostRegressor(
    iterations=500, learning_rate=0.03, depth=7,
    l2_leaf_reg=3, random_state=42, verbose=0
)
cat_A.fit(X_A, y_A)

preds_A_49  = (lgb_A.predict(df_49[features_gbdt]) + xgb_A.predict(df_49[features_gbdt]) + cat_A.predict(df_49[features_gbdt])) / 3.0
preds_A_test = (lgb_A.predict(test_df[features_gbdt]) + xgb_A.predict(test_df[features_gbdt]) + cat_A.predict(test_df[features_gbdt])) / 3.0

# Calibrate Model A
lr_cal = LinearRegression()
lr_cal.fit(preds_A_49.reshape(-1, 1), df_49[target].values)
preds_A_49_cal = lr_cal.predict(preds_A_49.reshape(-1, 1)).ravel()
preds_A_test_cal = lr_cal.predict(preds_A_test.reshape(-1, 1)).ravel()
r2_A_49 = r2_score(df_49[target], preds_A_49_cal)
print(f"  Model A R2 on Day 49 (calibrated): {r2_A_49:.4f}")

# Model B: Train on Day 49 ONLY
print("  Training GBDT Model B (Day 49 only)...")
X_B = df_49[features_gbdt]
y_B = df_49[target]

lgb_B = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=10, reg_alpha=0.5, reg_lambda=0.5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1, n_jobs=-1
)
lgb_B.fit(X_B, y_B)

xgb_B = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    reg_lambda=2.0, reg_alpha=0.5, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0, n_jobs=-1
)
xgb_B.fit(X_B, y_B)

cat_B = CatBoostRegressor(
    iterations=300, learning_rate=0.05, depth=6,
    l2_leaf_reg=5, random_state=42, verbose=0
)
cat_B.fit(X_B, y_B)

preds_B_test = (lgb_B.predict(test_df[features_gbdt]) + xgb_B.predict(test_df[features_gbdt]) + cat_B.predict(test_df[features_gbdt])) / 3.0
preds_B_49   = (lgb_B.predict(df_49[features_gbdt]) + xgb_B.predict(df_49[features_gbdt]) + cat_B.predict(df_49[features_gbdt])) / 3.0
r2_B_49 = r2_score(df_49[target], preds_B_49)
print(f"  Model B R2 on Day 49 (in-sample): {r2_B_49:.4f}")

# -- 3-WAY BLEND SEARCH: Model A (cal) + Model B + NN --
print("\n  Searching for optimal 3-way blend...")

best_blend_r2 = -1
best_wA, best_wB, best_wNN = 0, 0, 0

for wNN in np.arange(0.0, 1.01, 0.05):
    for wB in np.arange(0.0, 1.01 - wNN, 0.05):
        wA = 1.0 - wNN - wB
        if wA < -0.01:
            continue
        blended = wA * preds_A_49_cal + wB * preds_B_49 + wNN * nn_preds_d49_avg
        r2 = r2_score(df_49[target], blended)
        if r2 > best_blend_r2:
            best_blend_r2 = r2
            best_wA, best_wB, best_wNN = wA, wB, wNN

print(f"  Best 3-way blend: {best_wA:.2f}*A + {best_wB:.2f}*B + {best_wNN:.2f}*NN -> R2={best_blend_r2:.4f}")

# Also search 2-way: GBDT_best vs NN
# First find best GBDT-only blend
best_gbdt_r2 = -1
best_gbdt_wB = 0
for w in np.arange(0.0, 1.01, 0.05):
    blended = w * preds_B_49 + (1 - w) * preds_A_49_cal
    r2 = r2_score(df_49[target], blended)
    if r2 > best_gbdt_r2:
        best_gbdt_r2 = r2
        best_gbdt_wB = w

gbdt_best_d49 = best_gbdt_wB * preds_B_49 + (1 - best_gbdt_wB) * preds_A_49_cal
gbdt_best_test = best_gbdt_wB * preds_B_test + (1 - best_gbdt_wB) * preds_A_test_cal

best_nn_gbdt_r2 = -1
best_nn_w = 0
for w in np.arange(0.0, 1.01, 0.05):
    blended = w * nn_preds_d49_avg + (1 - w) * gbdt_best_d49
    r2 = r2_score(df_49[target], blended)
    if r2 > best_nn_gbdt_r2:
        best_nn_gbdt_r2 = r2
        best_nn_w = w

print(f"  Best GBDT-only blend: {best_gbdt_wB:.2f}*B + {1-best_gbdt_wB:.2f}*A -> R2={best_gbdt_r2:.4f}")
print(f"  Best NN+GBDT blend:  {best_nn_w:.2f}*NN + {1-best_nn_w:.2f}*GBDT_best -> R2={best_nn_gbdt_r2:.4f}")


# ---------------------------------------------------------------------
# GENERATE SUBMISSIONS
# ---------------------------------------------------------------------
print("\n  Generating submissions...")

submissions = []

# 1. Pure NN
preds_nn = np.clip(nn_preds_test_avg, 0.0, 1.0)
submissions.append(("submission_nn.csv", preds_nn, "Pure NN ensemble"))

# 2. Best GBDT only (same as original pipeline)
preds_gbdt = np.clip(gbdt_best_test, 0.0, 1.0)
submissions.append(("submission_gbdt.csv", preds_gbdt, f"GBDT ({best_gbdt_wB:.0%}B+{1-best_gbdt_wB:.0%}A)"))

# 3. Best 3-way blend
preds_3way = np.clip(best_wA * preds_A_test_cal + best_wB * preds_B_test + best_wNN * nn_preds_test_avg, 0.0, 1.0)
submissions.append(("submission_3way_blend.csv", preds_3way, f"3-way ({best_wA:.0%}A+{best_wB:.0%}B+{best_wNN:.0%}NN)"))

# 4. Best NN+GBDT 2-way blend
preds_2way = np.clip(best_nn_w * nn_preds_test_avg + (1 - best_nn_w) * gbdt_best_test, 0.0, 1.0)
submissions.append(("submission_nn_gbdt_blend.csv", preds_2way, f"NN+GBDT ({best_nn_w:.0%}NN+{1-best_nn_w:.0%}GBDT)"))

# 5. Fixed blends for experimentation
for nn_frac in [0.1, 0.2, 0.3]:
    preds_fixed = np.clip(nn_frac * nn_preds_test_avg + (1 - nn_frac) * gbdt_best_test, 0.0, 1.0)
    submissions.append((f"submission_{int(nn_frac*100)}nn_{int((1-nn_frac)*100)}gbdt.csv", 
                        preds_fixed, f"{nn_frac:.0%} NN + {1-nn_frac:.0%} GBDT"))

for fname, preds, desc in submissions:
    sub = pd.DataFrame({'Index': test['Index'].values, 'demand': preds})
    sub.to_csv(fname, index=False)
    print(f"  {fname:40s} | {desc:35s} | mean={preds.mean():.4f}, std={preds.std():.4f}")


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------
print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  NN Ensemble R2 on Day 49:          {nn_r2_d49:.4f}")
print(f"  GBDT Model A R2 (Day 49 cal):      {r2_A_49:.4f}")
print(f"  GBDT Model B R2 (Day 49 in-sample):{r2_B_49:.4f}")
print(f"  Best GBDT blend R2 (Day 49):       {best_gbdt_r2:.4f}")
print(f"  Best 3-way blend R2 (Day 49):      {best_blend_r2:.4f}")
print(f"  Best NN+GBDT blend R2 (Day 49):    {best_nn_gbdt_r2:.4f}")
print()
print("  Recommended submission order:")
print("    1. submission_3way_blend.csv      (highest Day 49 R2)")
print("    2. submission_nn_gbdt_blend.csv   (NN+GBDT 2-way)")
print("    3. submission_gbdt.csv            (GBDT baseline)")
print("    4. submission_nn.csv              (pure NN)")
print("=" * 70)

# Cleanup
import os
if os.path.exists('_check_data.py'):
    os.remove('_check_data.py')
