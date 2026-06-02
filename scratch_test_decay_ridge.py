import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

train = pd.read_csv('dataset/train.csv')
df_48 = train[train['day'] == 48].copy()
df_49 = train[train['day'] == 49].copy()

# Parse time
for df in [df_48, df_49]:
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['sin_time'] = np.sin(2 * np.pi * df['time_minutes'] / 1440)
    df['cos_time'] = np.cos(2 * np.pi * df['time_minutes'] / 1440)
    df['geo_p4'] = df['geohash'].str[:4]
    df['geo_p5'] = df['geohash'].str[:5]

# Impute categories
road_type_map = train.groupby('geohash')['RoadType'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_road_type = train['RoadType'].mode()[0]
for df in [df_48, df_49]:
    df['RoadType'] = df['RoadType'].fillna(df['geohash'].map(road_type_map)).fillna(global_road_type)

lanes_map = train.groupby('geohash')['NumberofLanes'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lanes = train['NumberofLanes'].mode()[0]
for df in [df_48, df_49]:
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['geohash'].map(lanes_map)).fillna(global_lanes)

lv_map = train.groupby('geohash')['LargeVehicles'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lv = train['LargeVehicles'].mode()[0]
for df in [df_48, df_49]:
    df['LargeVehicles'] = df['LargeVehicles'].fillna(df['geohash'].map(lv_map)).fillna(global_lv)

lm_map = train.groupby('geohash')['Landmarks'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lm = train['Landmarks'].mode()[0]
for df in [df_48, df_49]:
    df['Landmarks'] = df['Landmarks'].fillna(df['geohash'].map(lm_map)).fillna(global_lm)

geo_temp = train.groupby('geohash')['Temperature'].mean().to_dict()
ts_temp = train.groupby('timestamp')['Temperature'].mean().to_dict()
global_temp = train['Temperature'].mean()
for df in [df_48, df_49]:
    df['Temperature'] = df['Temperature'].fillna(df['geohash'].map(geo_temp)).fillna(df['timestamp'].map(ts_temp)).fillna(global_temp)

ts_weather = train.groupby('timestamp')['Weather'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_weather = train['Weather'].mode()[0]
for df in [df_48, df_49]:
    df['Weather'] = df['Weather'].fillna(df['timestamp'].map(ts_weather)).fillna(global_weather)

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
                if bit:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon

for df in [df_48, df_49]:
    lats, lons = [], []
    for g in df['geohash']:
        lat, lon = decode_geohash(g)
        lats.append(lat)
        lons.append(lon)
    df['lat'] = lats
    df['lon'] = lons

features_A = [
    'lat', 'lon', 'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time',
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    'geo_p4', 'geo_p5', 'geohash'
]

cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geo_p4', 'geo_p5', 'geohash']

# Early morning shift metrics on Day 48
early_ts = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
df_48_early = df_48[df_48['timestamp'].isin(early_ts)]
mean_demand_48_early = df_48_early.groupby('geohash')['demand'].mean().to_dict()
global_mean_48_early = df_48_early['demand'].mean()

day48_ts_lookup = df_48.set_index(['geohash', 'time_minutes'])['demand'].to_dict()
geo_mean_48 = df_48.groupby('geohash')['demand'].mean().to_dict()
global_mean_48 = df_48['demand'].mean()

# Run K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df_49['pred_A_oof'] = 0.0

print("Generating OOF predictions for Model A...")
for tr_idx, val_idx in kf.split(df_49):
    tr = df_49.iloc[tr_idx]
    val = df_49.iloc[val_idx].copy()
    
    train_fold = pd.concat([df_48, tr], ignore_index=True)
    
    for col in cat_cols:
        train_fold[col] = train_fold[col].astype('category')
        val[col] = val[col].astype('category')
        
    lgb_A = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1)
    lgb_A.fit(train_fold[features_A], train_fold['demand'])
    pred_lgb = lgb_A.predict(val[features_A])
    
    xgb_A = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, random_state=42, verbosity=0, enable_categorical=True)
    xgb_A.fit(train_fold[features_A], train_fold['demand'])
    pred_xgb = xgb_A.predict(val[features_A])
    
    cat_A = CatBoostRegressor(iterations=150, learning_rate=0.08, random_state=42, verbose=0, cat_features=cat_cols)
    cat_A.fit(train_fold[features_A], train_fold['demand'])
    pred_cat = cat_A.predict(val[features_A])
    
    pred_A_val = (pred_lgb + pred_xgb + pred_cat) / 3.0
    df_49.iloc[val_idx, df_49.columns.get_loc('pred_A_oof')] = pred_A_val

# Add raw historical feature
df_49['prev_day_demand'] = df_49.apply(lambda r: day48_ts_lookup.get((r['geohash'], r['time_minutes']), np.nan), axis=1)
df_49['geo_mean_demand'] = df_49['geohash'].map(geo_mean_48).fillna(global_mean_48)
df_49['prev_day_demand'] = df_49['prev_day_demand'].fillna(df_49['geo_mean_demand']).astype(float)

# Evaluate Ridge model in CV
scores_ridge = []
for tr_idx, val_idx in kf.split(df_49):
    tr = df_49.iloc[tr_idx]
    val = df_49.iloc[val_idx].copy()
    
    # Calculate OOF early means
    tr_early_mean = tr.groupby('geohash')['demand'].mean().to_dict()
    global_tr_early_mean = tr['demand'].mean()
    
    # Recalculate features
    tr_copy = tr.copy()
    val['mean_demand_48_early'] = val['geohash'].map(mean_demand_48_early).fillna(global_mean_48_early)
    val['mean_demand_49_early'] = val['geohash'].map(tr_early_mean).fillna(global_tr_early_mean)
    val['early_diff'] = val['mean_demand_49_early'] - val['mean_demand_48_early']
    
    tr_copy['mean_demand_48_early'] = tr_copy['geohash'].map(mean_demand_48_early).fillna(global_mean_48_early)
    tr_copy['mean_demand_49_early'] = tr_copy['geohash'].map(tr_early_mean).fillna(global_tr_early_mean)
    tr_copy['early_diff'] = tr_copy['mean_demand_49_early'] - tr_copy['mean_demand_48_early']
    
    # decay factor
    mean_decay_diff = np.mean([max(0.0, 1.0 - t/197.6) for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]])
    
    val['decay_factor'] = val['time_minutes'].apply(lambda t: max(0.0, 1.0 - t/197.6)) / mean_decay_diff
    val['decay_shift'] = val['early_diff'] * val['decay_factor']
    
    tr_copy['decay_factor'] = tr_copy['time_minutes'].apply(lambda t: max(0.0, 1.0 - t/197.6)) / mean_decay_diff
    tr_copy['decay_shift'] = tr_copy['early_diff'] * tr_copy['decay_factor']
    
    # Ridge
    f_meta = ['pred_A_oof', 'prev_day_demand', 'decay_shift']
    meta = Ridge(alpha=1.0)
    meta.fit(tr_copy[f_meta], tr_copy['demand'])
    
    preds_val = meta.predict(val[f_meta])
    scores_ridge.append(r2_score(val['demand'], np.clip(preds_val, 0, 1)))

print(f"CV R2 for Decaying Shift Ridge Stacking: {np.mean(scores_ridge):.5f}")

# Fit final Ridge model and print coefficients
df_49_copy = df_49.copy()
mean_demand_49_early = df_49.groupby('geohash')['demand'].mean().to_dict()
global_mean_49_early = df_49['demand'].mean()

df_49_copy['mean_demand_48_early'] = df_49_copy['geohash'].map(mean_demand_48_early).fillna(global_mean_48_early)
df_49_copy['mean_demand_49_early'] = df_49_copy['geohash'].map(mean_demand_49_early).fillna(global_mean_49_early)
df_49_copy['early_diff'] = df_49_copy['mean_demand_49_early'] - df_49_copy['mean_demand_48_early']

mean_decay_diff = np.mean([max(0.0, 1.0 - t/197.6) for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]])
df_49_copy['decay_factor'] = df_49_copy['time_minutes'].apply(lambda t: max(0.0, 1.0 - t/197.6)) / mean_decay_diff
df_49_copy['decay_shift'] = df_49_copy['early_diff'] * df_49_copy['decay_factor']

f_meta = ['pred_A_oof', 'prev_day_demand', 'decay_shift']
meta_final = Ridge(alpha=1.0)
meta_final.fit(df_49_copy[f_meta], df_49_copy['demand'])

print("\nFinal Ridge Coefficients:")
for f, c in zip(f_meta, meta_final.coef_):
    print(f"{f:20s}: {c:+.4f}")
print(f"Intercept: {meta_final.intercept_:.4f}")
