import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

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

# First, generate OOF predictions for Model A on df_49_train
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df_49['pred_A_oof'] = 0.0

print("Generating OOF predictions for Model A on Day 49...")
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

# Now run a second layer CV to evaluate the OOF Calibration R2
scores_oof_cal = []
for tr_idx, val_idx in kf.split(df_49):
    tr = df_49.iloc[tr_idx]
    val = df_49.iloc[val_idx]
    
    lr = LinearRegression()
    lr.fit(tr['pred_A_oof'].values.reshape(-1, 1), tr['demand'].values)
    pred_cal = lr.predict(val['pred_A_oof'].values.reshape(-1, 1))
    scores_oof_cal.append(r2_score(val['demand'], np.clip(pred_cal, 0, 1)))
    
print(f"CV R2 for OOF 1D Linear Calibration: {np.mean(scores_oof_cal):.5f}")

# Train the final calibration model on all of df_49
lr_final = LinearRegression()
lr_final.fit(df_49['pred_A_oof'].values.reshape(-1, 1), df_49['demand'].values)
print(f"Final OOF Calibration Slope: {lr_final.coef_[0]:.4f}, Intercept: {lr_final.intercept_:.4f}")
