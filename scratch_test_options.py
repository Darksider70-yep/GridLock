import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

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

cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geo_p4', 'geo_p5', 'geohash']
for col in cat_cols:
    df_48[col] = df_48[col].astype('category')
    df_49[col] = df_49[col].astype('category')

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

print("Training Model A components ONLY on Day 48...")
lgb_A = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, random_state=42, verbose=-1)
lgb_A.fit(df_48[features_A], df_48['demand'])

xgb_A = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, random_state=42, verbosity=0, enable_categorical=True)
xgb_A.fit(df_48[features_A], df_48['demand'])

cat_A = CatBoostRegressor(iterations=300, learning_rate=0.04, random_state=42, verbose=0, cat_features=cat_cols)
cat_A.fit(df_48[features_A], df_48['demand'])

df_49['pred_A'] = (lgb_A.predict(df_49[features_A]) + xgb_A.predict(df_49[features_A]) + cat_A.predict(df_49[features_A])) / 3.0

# Setup shift metrics
early_ts = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
df_48_early = df_48[df_48['timestamp'].isin(early_ts)]
mean_demand_48_early = df_48_early.groupby('geohash')['demand'].mean().to_dict()
global_mean_48_early = df_48_early['demand'].mean()

mean_demand_49_early = df_49.groupby('geohash')['demand'].mean().to_dict()
global_mean_49_early = df_49['demand'].mean()

df_49['mean_48_early'] = df_49['geohash'].map(mean_demand_48_early).fillna(global_mean_48_early)
df_49['mean_49_early'] = df_49['geohash'].map(mean_demand_49_early).fillna(global_mean_49_early)
df_49['early_diff'] = df_49['mean_49_early'] - df_49['mean_48_early']
df_49['early_ratio'] = df_49['mean_49_early'] / (df_49['mean_48_early'] + 1e-3)

# Evaluate Options on df_49
# 1. No Shift
print("\n1. No Shift R2:                     ", r2_score(df_49['demand'], df_49['pred_A']))

# 2. Constant Additive Shift
df_49['pred_const_add'] = df_49['pred_A'] + df_49['early_diff']
print("2. Constant Additive Shift R2:      ", r2_score(df_49['demand'], np.clip(df_49['pred_const_add'], 0, 1)))

# 3. Constant Multiplicative Shift
df_49['pred_const_mult'] = df_49['pred_A'] * df_49['early_ratio']
print("3. Constant Multiplicative Shift R2:", r2_score(df_49['demand'], np.clip(df_49['pred_const_mult'], 0, 1)))

# 4. Decaying Additive Shift (decays to 0 by t=197.6)
# shift(t) = early_diff * max(0, 1.0 - t/197.6)
mean_decay_diff = np.mean([max(0.0, 1.0 - t/197.6) for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]])
df_49['decay_factor_diff'] = df_49['time_minutes'].apply(lambda t: max(0.0, 1.0 - t/197.6)) / mean_decay_diff
df_49['pred_decay_add'] = df_49['pred_A'] + df_49['early_diff'] * df_49['decay_factor_diff']
print("4. Decaying Additive Shift R2:      ", r2_score(df_49['demand'], np.clip(df_49['pred_decay_add'], 0, 1)))

# 5. Decaying Multiplicative Shift (decays to 1.0 by t=160.8)
# ratio(t) = 1.0 + (early_ratio - 1.0) * max(0, 1.0 - t/160.8)
mean_decay_ratio = np.mean([max(0.0, 1.0 - t/160.8) for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]])
# We scale the excess ratio early_ratio - 1.0
df_49['decay_factor_ratio'] = df_49['time_minutes'].apply(lambda t: max(0.0, 1.0 - t/160.8)) / (mean_decay_ratio + 1e-5)
df_49['pred_decay_mult'] = df_49['pred_A'] * (1.0 + (df_49['early_ratio'] - 1.0) * df_49['decay_factor_ratio'])
print("5. Decaying Multiplicative Shift R2:", r2_score(df_49['demand'], np.clip(df_49['pred_decay_mult'], 0, 1)))
