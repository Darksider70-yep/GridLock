import pandas as pd
import numpy as np

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print('=== TRAIN DATA ===')
print(train.shape)
print(train.columns.tolist())
print(train.dtypes)
print()
print(train.head(3).to_string())
print()
print('=== Key Statistics ===')
days = train['day'].unique()
print(f'Days in train: {days}')
n_geo = train['geohash'].nunique()
print(f'Unique geohashes: {n_geo}')
n_ts = train['timestamp'].nunique()
print(f'Unique timestamps: {n_ts}')
print(f'Rows per day: {len(train) / len(days):.0f}')
print(f'Demand range: [{train["demand"].min():.4f}, {train["demand"].max():.4f}]')
print(f'Demand mean: {train["demand"].mean():.4f}')
print(f'Demand std: {train["demand"].std():.4f}')
print()
print('Missing values per column:')
print(train.isnull().sum())
print()

print('=== TEST DATA ===')
print(test.shape)
test_days = test['day'].unique()
print(f'Days in test: {test_days}')
n_ts_test = test['timestamp'].nunique()
print(f'Total unique timestamps in test: {n_ts_test}')
test_timestamps = sorted(test['timestamp'].unique())
print(f'Test timestamps (first 10): {test_timestamps[:10]}')
print(f'Test timestamps (last 10): {test_timestamps[-10:]}')
print()
print('Missing values per column:')
print(test.isnull().sum())

print()
print('=== DAY-LEVEL DEMAND ANALYSIS ===')
for d in sorted(train['day'].unique()):
    dd = train[train['day'] == d]['demand']
    print(f'Day {d}: mean={dd.mean():.4f}, std={dd.std():.4f}, count={len(dd)}')

print()
print('=== CATEGORICAL VALUE COUNTS ===')
for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    print(f'\n{col}:')
    print(train[col].value_counts(dropna=False).head(10))

print()
print('=== GEOHASH OVERLAP ===')
train_geos = set(train['geohash'].unique())
test_geos = set(test['geohash'].unique())
print(f'Train geohashes: {len(train_geos)}')
print(f'Test geohashes: {len(test_geos)}')
print(f'Overlap: {len(train_geos & test_geos)}')
print(f'Test-only: {len(test_geos - train_geos)}')

print()
print('=== TIMESTAMP OVERLAP ===')
train_ts = set(train['timestamp'].unique())
test_ts = set(test['timestamp'].unique())
print(f'Train timestamps: {len(train_ts)}')
print(f'Test timestamps: {len(test_ts)}')
print(f'Overlap: {len(train_ts & test_ts)}')

print()
print('=== DAY 48 vs DAY 49 DETAILED DEMAND COMPARISON ===')
d48 = train[train['day'] == 48]
d49 = train[train['day'] == 49]

# Per-timestamp comparison
ts_mean_48 = d48.groupby('timestamp')['demand'].mean()
ts_mean_49 = d49.groupby('timestamp')['demand'].mean()
common_ts = ts_mean_48.index.intersection(ts_mean_49.index)
if len(common_ts) > 0:
    diff = ts_mean_49[common_ts] - ts_mean_48[common_ts]
    print(f'Avg demand shift (D49-D48) across timestamps: {diff.mean():.4f}')
    print(f'Std of shift: {diff.std():.4f}')
    print(f'Max shift: {diff.max():.4f}')
    print(f'Min shift: {diff.min():.4f}')

# Per-geohash comparison
geo_mean_48 = d48.groupby('geohash')['demand'].mean()
geo_mean_49 = d49.groupby('geohash')['demand'].mean()
common_geo = geo_mean_48.index.intersection(geo_mean_49.index)
geo_diff = geo_mean_49[common_geo] - geo_mean_48[common_geo]
print(f'\nPer-geohash avg demand shift: {geo_diff.mean():.4f}')
print(f'Per-geohash shift std: {geo_diff.std():.4f}')

# Correlation between Day 48 and Day 49 at geohash-timestamp level
merged = d48.set_index(['geohash', 'timestamp'])['demand'].rename('d48')
merged = merged.to_frame().join(d49.set_index(['geohash', 'timestamp'])['demand'].rename('d49'))
merged = merged.dropna()
from sklearn.metrics import r2_score
r2 = r2_score(merged['d49'], merged['d48'])
print(f'\nR2 of predicting Day49 demand directly from Day48 demand (same geo+ts): {r2:.4f}')

# Just using Day 48 as-is
print(f'Correlation between D48 and D49 demand: {merged["d48"].corr(merged["d49"]):.4f}')
