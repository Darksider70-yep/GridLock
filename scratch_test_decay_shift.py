import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

train = pd.read_csv('dataset/train.csv')
df_48 = train[train['day'] == 48].copy()
df_49 = train[train['day'] == 49].copy()

# Parse time
for df in [df_48, df_49]:
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_minutes'] = df['hour'] * 60 + df['minute']

early_ts = ['0:0', '0:15', '0:30', '0:45', '1:0', '1:15', '1:30', '1:45', '2:0']
df_49_early = df_49[df_49['timestamp'].isin(early_ts)].copy()

# Day 48 lookup
day48_ts_lookup = df_48.set_index(['geohash', 'time_minutes'])['demand'].to_dict()

# Base mean
geo_mean_48 = df_48.groupby('geohash')['demand'].mean().to_dict()
global_mean_48 = df_48['demand'].mean()

df_49_early['prev_day_demand'] = df_49_early.apply(lambda r: day48_ts_lookup.get((r['geohash'], r['time_minutes']), np.nan), axis=1)
df_49_early['geo_mean_demand'] = df_49_early['geohash'].map(geo_mean_48).fillna(global_mean_48)
df_49_early['prev_day_demand'] = df_49_early['prev_day_demand'].fillna(df_49_early['geo_mean_demand'])

# Let's define the decaying shift function
# diff(t) = 0.051155 - 0.000258 * t
# We can also make it geohash-specific or global. Let's start with global decaying shift.
df_49_early['decay_shift'] = 0.051155 - 0.000258 * df_49_early['time_minutes']
df_49_early['pred_decay'] = df_49_early['prev_day_demand'] + df_49_early['decay_shift']

print("R2 score of Global Decaying Shift:   ", r2_score(df_49_early['demand'], np.clip(df_49_early['pred_decay'], 0, 1)))

# Let's compare with a constant global shift (+0.0376)
df_49_early['pred_const'] = df_49_early['prev_day_demand'] + 0.037647
print("R2 score of Constant Global Shift:   ", r2_score(df_49_early['demand'], np.clip(df_49_early['pred_const'], 0, 1)))

# Let's check geohash-specific decaying shift!
# We can estimate a slope and intercept for each geohash:
# intercept_g = early_diff_0_g (difference at 0:00)
# slope_g = (early_diff_120_g - early_diff_0_g) / 120
# To avoid noise, let's use the geohash early morning mean difference and apply the global decay shape:
# shift_g(t) = early_diff_g * (decay_shift(t) / mean_decay_shift)
# Where mean_decay_shift is the average of decay_shift(t) over 0 to 120 minutes.
mean_decay_shift = np.mean([0.051155 - 0.000258 * t for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]])

# Let's compute geohash-level early_diff
df_48_early = df_48[df_48['timestamp'].isin(early_ts)]
mean_48_early = df_48_early.groupby('geohash')['demand'].mean().to_dict()
mean_49_early = df_49_early.groupby('geohash')['demand'].mean().to_dict()
global_48_early = df_48_early['demand'].mean()
global_49_early = df_49_early['demand'].mean()

df_49_early['mean_48_early'] = df_49_early['geohash'].map(mean_48_early).fillna(global_48_early)
df_49_early['mean_49_early'] = df_49_early['geohash'].map(mean_49_early).fillna(global_49_early)
df_49_early['early_diff'] = df_49_early['mean_49_early'] - df_49_early['mean_48_early']

# Geohash-level decaying shift: early_diff_g * (decay_shift(t) / mean_decay_shift)
df_49_early['geo_decay_shift'] = df_49_early['early_diff'] * (df_49_early['decay_shift'] / mean_decay_shift)
df_49_early['pred_geo_decay'] = df_49_early['prev_day_demand'] + df_49_early['geo_decay_shift']

print("R2 score of Geohash Decaying Shift:  ", r2_score(df_49_early['demand'], np.clip(df_49_early['pred_geo_decay'], 0, 1)))

# Constant geohash shift (which was 0.9152)
df_49_early['pred_geo_const'] = df_49_early['prev_day_demand'] + df_49_early['early_diff']
print("R2 score of Constant Geohash Shift:  ", r2_score(df_49_early['demand'], np.clip(df_49_early['pred_geo_const'], 0, 1)))
