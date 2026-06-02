import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()

print('=== CRITICAL INSIGHT: Day 49 in train only has EARLY morning data ===')
d49_ts = sorted(d49['timestamp'].unique())
print(f'Day 49 train timestamps ({len(d49_ts)}): {d49_ts}')
print(f'Day 49 train rows: {len(d49)}')
print(f'Day 48 train rows: {len(d48)}')
print()

# What timestamps are in test?
test_ts = sorted(test['timestamp'].unique())
print(f'Test timestamps ({len(test_ts)}):')
# Parse and sort by time_minutes
def ts_to_minutes(ts):
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

test_ts_sorted = sorted(test_ts, key=ts_to_minutes)
print(test_ts_sorted)
print()

# What hours?
test_hours = sorted(set([ts_to_minutes(t) // 60 for t in test_ts]))
print(f'Test hours: {test_hours}')

d49_hours = sorted(set([ts_to_minutes(t) // 60 for t in d49_ts]))
print(f'Day 49 train hours: {d49_hours}')
print()

# So test is Day 49, hours 2:15 onwards through 23:45
# And Day 49 train is hours 0:0 through 2:0

# Check the shift pattern by hour
d48['hour'] = d48['timestamp'].apply(lambda x: int(x.split(':')[0]))
d49['hour'] = d49['timestamp'].apply(lambda x: int(x.split(':')[0]))

print('=== HOUR-LEVEL Day 48 stats ===')
for h in sorted(d48['hour'].unique()):
    dd = d48[d48['hour'] == h]['demand']
    print(f'Hour {h:2d}: mean={dd.mean():.4f}, std={dd.std():.4f}, n={len(dd)}')

print()
print('=== HOUR-LEVEL Day 49 train stats ===')
for h in sorted(d49['hour'].unique()):
    dd = d49[d49['hour'] == h]['demand']
    print(f'Hour {h:2d}: mean={dd.mean():.4f}, std={dd.std():.4f}, n={len(dd)}')

print()
print('=== GEOHASH-LEVEL ANALYSIS ===')
# How many geohashes appear in Day 49 train vs test
d49_geo = set(d49['geohash'].unique())
test_geo = set(test['geohash'].unique())
d48_geo = set(d48['geohash'].unique())
print(f'Day 48 geohashes: {len(d48_geo)}')
print(f'Day 49 train geohashes: {len(d49_geo)}')
print(f'Test geohashes: {len(test_geo)}')
print(f'D49 train & test overlap: {len(d49_geo & test_geo)}')
print(f'D48 & test overlap: {len(d48_geo & test_geo)}')

print()
print('=== CHECKING WHETHER EVERY GEOHASH HAS ALL TIMESTAMPS ===')
d48_per_geo = d48.groupby('geohash').size()
print(f'Day 48 timestamps per geohash: min={d48_per_geo.min()}, max={d48_per_geo.max()}, mean={d48_per_geo.mean():.1f}')
d49_per_geo = d49.groupby('geohash').size()
print(f'Day 49 train timestamps per geohash: min={d49_per_geo.min()}, max={d49_per_geo.max()}, mean={d49_per_geo.mean():.1f}')

# Check if test has consistent number of timestamps per geohash
test_per_geo = test.groupby('geohash').size()
print(f'Test timestamps per geohash: min={test_per_geo.min()}, max={test_per_geo.max()}, mean={test_per_geo.mean():.1f}')

# Many geohashes NOT having all timestamps in Day 48
sparse_geos_48 = (d48_per_geo < 96).sum()
print(f'\nDay 48 geohashes with < 96 timestamps: {sparse_geos_48}')
full_geos_48 = (d48_per_geo == 96).sum()
print(f'Day 48 geohashes with exactly 96 timestamps: {full_geos_48}')

print()
print('=== DEMAND DISTRIBUTION CHARACTERISTICS ===')
print(f'Mean: {train["demand"].mean():.4f}')
print(f'Median: {train["demand"].median():.4f}')
print(f'Pct zeros: {(train["demand"] == 0).mean() * 100:.1f}%')
print(f'Pct < 0.01: {(train["demand"] < 0.01).mean() * 100:.1f}%')
print(f'Pct > 0.5: {(train["demand"] > 0.5).mean() * 100:.1f}%')
print(f'Pct > 0.8: {(train["demand"] > 0.8).mean() * 100:.1f}%')

# Temperature correlation 
print()
print('=== TEMPERATURE ANALYSIS ===')
temp_corr = train[['Temperature', 'demand']].dropna().corr()
print(f'Temperature-demand correlation: {temp_corr.loc["Temperature", "demand"]:.4f}')

# Weather effect
print()
print('=== WEATHER EFFECT ON DEMAND ===')
for w in ['Sunny', 'Rainy', 'Foggy', 'Snowy']:
    dd = train[train['Weather'] == w]['demand']
    print(f'{w:6s}: mean={dd.mean():.4f}, std={dd.std():.4f}, n={len(dd)}')

# NumberofLanes effect
print()
print('=== NUMBER OF LANES EFFECT ===')
for n in sorted(train['NumberofLanes'].unique()):
    dd = train[train['NumberofLanes'] == n]['demand']
    print(f'Lanes {n}: mean={dd.mean():.4f}, std={dd.std():.4f}, n={len(dd)}')

# RoadType effect
print()
print('=== ROAD TYPE EFFECT ===')
for r in train['RoadType'].dropna().unique():
    dd = train[train['RoadType'] == r]['demand']
    print(f'{r:12s}: mean={dd.mean():.4f}, std={dd.std():.4f}, n={len(dd)}')

print()
print('=== DEMAND AUTO-CORRELATION TEST ===')
# For same geohash, how well does demand at t-15 predict demand at t?
d48_sorted = d48.sort_values(['geohash', 'timestamp'])
d48_sorted['time_minutes'] = d48_sorted['timestamp'].apply(ts_to_minutes)
d48_sorted = d48_sorted.sort_values(['geohash', 'time_minutes'])
d48_sorted['demand_lag1'] = d48_sorted.groupby('geohash')['demand'].shift(1)
valid = d48_sorted.dropna(subset=['demand_lag1'])
r2_lag = r2_score(valid['demand'], valid['demand_lag1'])
corr_lag = valid['demand'].corr(valid['demand_lag1'])
print(f'R2 (demand ~ demand_lag15): {r2_lag:.4f}')
print(f'Corr (demand, demand_lag15): {corr_lag:.4f}')

# 2-step lag (30 min)
d48_sorted['demand_lag2'] = d48_sorted.groupby('geohash')['demand'].shift(2)
valid2 = d48_sorted.dropna(subset=['demand_lag2'])
r2_lag2 = r2_score(valid2['demand'], valid2['demand_lag2'])
print(f'R2 (demand ~ demand_lag30): {r2_lag2:.4f}')

# 4-step lag (60 min)
d48_sorted['demand_lag4'] = d48_sorted.groupby('geohash')['demand'].shift(4)
valid4 = d48_sorted.dropna(subset=['demand_lag4'])
r2_lag4 = r2_score(valid4['demand'], valid4['demand_lag4'])
print(f'R2 (demand ~ demand_lag60): {r2_lag4:.4f}')
