import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

df_all = pd.concat([train, test], ignore_index=True)

# Parse time
df_all['hour'] = df_all['timestamp'].apply(lambda x: int(x.split(':')[0]))
df_all['minute'] = df_all['timestamp'].apply(lambda x: int(x.split(':')[1]))
df_all['time_minutes'] = df_all['hour'] * 60 + df_all['minute']
df_all['sin_time'] = np.sin(2 * np.pi * df_all['time_minutes'] / 1440)
df_all['cos_time'] = np.cos(2 * np.pi * df_all['time_minutes'] / 1440)

# Decoding geohashes
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

lats, lons = [], []
for g in df_all['geohash']:
    lat, lon = decode_geohash(g)
    lats.append(lat)
    lons.append(lon)
df_all['lat'] = lats
df_all['lon'] = lons

df_all['geo_p4'] = df_all['geohash'].str[:4]
df_all['geo_p5'] = df_all['geohash'].str[:5]

# Imputation
road_type_map = train.groupby('geohash')['RoadType'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_road_type = train['RoadType'].mode()[0]
df_all['RoadType'] = df_all['RoadType'].fillna(df_all['geohash'].map(road_type_map)).fillna(global_road_type)

lanes_map = train.groupby('geohash')['NumberofLanes'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lanes = train['NumberofLanes'].mode()[0]
df_all['NumberofLanes'] = df_all['NumberofLanes'].fillna(df_all['geohash'].map(lanes_map)).fillna(global_lanes)

lv_map = train.groupby('geohash')['LargeVehicles'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lv = train['LargeVehicles'].mode()[0]
df_all['LargeVehicles'] = df_all['LargeVehicles'].fillna(df_all['geohash'].map(lv_map)).fillna(global_lv)

lm_map = train.groupby('geohash')['Landmarks'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_lm = train['Landmarks'].mode()[0]
df_all['Landmarks'] = df_all['Landmarks'].fillna(df_all['geohash'].map(lm_map)).fillna(global_lm)

geo_temp = train.groupby('geohash')['Temperature'].mean().to_dict()
ts_temp = train.groupby('timestamp')['Temperature'].mean().to_dict()
global_temp = train['Temperature'].mean()
df_all['Temperature'] = df_all['Temperature'].fillna(df_all['geohash'].map(geo_temp)).fillna(df_all['timestamp'].map(ts_temp)).fillna(global_temp)

ts_weather = train.groupby('timestamp')['Weather'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
global_weather = train['Weather'].mode()[0]
df_all['Weather'] = df_all['Weather'].fillna(df_all['timestamp'].map(ts_weather)).fillna(global_weather)

# Label Encoding
cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geo_p4', 'geo_p5', 'geohash']
for col in cat_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

# Split back
train_df = df_all[df_all['demand'].notnull()].copy()
test_df = df_all[df_all['demand'].isnull()].copy()

df_48 = train_df[train_df['day'] == 48].copy()
df_49_train = train_df[train_df['day'] == 49].copy()

# Historical feature lookup
geo_mean_48 = df_48.groupby('geohash')['demand'].mean().to_dict()
global_mean_48 = df_48['demand'].mean()
day48_lookup = df_48.set_index(['geohash', 'timestamp'])['demand'].to_dict()
ts_mean_48 = df_48.groupby('timestamp')['demand'].mean().to_dict()

def add_geo_features(df):
    df['prev_day_demand'] = df.apply(lambda r: day48_lookup.get((r['geohash'], r['timestamp']), np.nan), axis=1)
    df['prev_day_demand'] = df['prev_day_demand'].fillna(df['geohash'].map(geo_mean_48))
    df['prev_day_demand'] = df['prev_day_demand'].fillna(df['timestamp'].map(ts_mean_48))
    df['prev_day_demand'] = df['prev_day_demand'].fillna(global_mean_48)
    df['geo_mean_demand'] = df['geohash'].map(geo_mean_48).fillna(global_mean_48)
    return df

train_df = add_geo_features(train_df)
test_df = add_geo_features(test_df)
df_49_train = add_geo_features(df_49_train)

features_A = [
    'lat', 'lon', 'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time',
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    'geo_p4', 'geo_p5', 'geohash', 'geo_mean_demand'
]
target = 'demand'

# Train Model A
lgb_A = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgb_A.fit(train_df[features_A], train_df[target])
preds_lgb_A_test = lgb_A.predict(test_df[features_A])
preds_lgb_A_train49 = lgb_A.predict(df_49_train[features_A])

xgb_A = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_A.fit(train_df[features_A], train_df[target])
preds_xgb_A_test = xgb_A.predict(test_df[features_A])
preds_xgb_A_train49 = xgb_A.predict(df_49_train[features_A])

cat_A = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
cat_A.fit(train_df[features_A], train_df[target])
preds_cat_A_test = cat_A.predict(test_df[features_A])
preds_cat_A_train49 = cat_A.predict(df_49_train[features_A])

preds_A_test = (preds_lgb_A_test + preds_xgb_A_test + preds_cat_A_test) / 3.0
preds_A_train49 = (preds_lgb_A_train49 + preds_xgb_A_train49 + preds_cat_A_train49) / 3.0

lr = LinearRegression()
lr.fit(preds_A_train49.reshape(-1, 1), df_49_train[target].values)
preds_A_test_cal = lr.predict(preds_A_test.reshape(-1, 1))

# Train Model B
features_B = features_A + ['prev_day_demand']
lgb_B = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgb_B.fit(df_49_train[features_B], df_49_train[target])
preds_lgb_B_test = lgb_B.predict(test_df[features_B])

xgb_B = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_B.fit(df_49_train[features_B], df_49_train[target])
preds_xgb_B_test = xgb_B.predict(test_df[features_B])

cat_B = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
cat_B.fit(df_49_train[features_B], df_49_train[target])
preds_cat_B_test = cat_B.predict(test_df[features_B])

preds_B_test = (preds_lgb_B_test + preds_xgb_B_test + preds_cat_B_test) / 3.0

# Final ensemble
final_preds = 0.8 * preds_B_test + 0.2 * preds_A_test_cal
final_preds = np.clip(final_preds, 0.0, 1.0)

print("\n--- Original Notebook Predictions Summary ---")
print(pd.Series(final_preds).describe())

print("\nModel B Predictions Summary:")
print(pd.Series(preds_B_test).describe())

print("\nModel A Calibrated Predictions Summary:")
print(pd.Series(preds_A_test_cal).describe())
