# ============================================================
# US House Price Prediction — Full Pipeline
# Data Prep → Treasury Features → Hyperparameter Tuning → Feature Selection
# ============================================================
# Paste into ONE Colab cell (Pro + High-RAM: 8 cores, ~55 GB RAM)
# Uploads: panel_features.csv AND DGS2.csv
# Outputs: downloads final files to your local machine
# Estimated runtime: ~1.5-3 hours
# ============================================================

import pandas as pd
import numpy as np
import os, time
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print(f"CPU cores: {os.cpu_count()}")
print(f"RAM: {__import__('psutil').virtual_memory().total / 1e9:.1f} GB")

# --- UPLOAD BOTH FILES ---
from google.colab import files
print("\nUpload BOTH files: panel_features.csv AND DGS2.csv")
uploaded = files.upload()
print(f"Uploaded: {list(uploaded.keys())}")


# ============================================================
# STAGE 1: REBUILD V2 PANEL (Treasury Features)
# ============================================================
print("\n" + "=" * 60)
print("STAGE 1: REBUILDING V2 PANEL WITH TREASURY FEATURES")
print("=" * 60)

# --- 10Y Treasury from yfinance ---
import yfinance as yf
print("\nPulling 10Y treasury yield...")
tnx = yf.download('^TNX', start='1996-01-01', end='2026-01-01', progress=False)
if isinstance(tnx.columns, pd.MultiIndex):
    tnx.columns = tnx.columns.get_level_values(0)
treasury_10y_q = tnx['Close'].resample('QE').mean()
treasury_10y_q.name = 'treasury_10y'
print(f"10Y: {len(treasury_10y_q)} quarters, "
      f"{treasury_10y_q.index.min().date()} to {treasury_10y_q.index.max().date()}")

# --- 2Y Treasury from FRED DGS2.csv ---
print("Loading 2Y yield from DGS2.csv...")
dgs2 = pd.read_csv('DGS2.csv', parse_dates=['observation_date'])
dgs2 = dgs2[dgs2['DGS2'] != '.'].copy()
dgs2['DGS2'] = dgs2['DGS2'].astype(float)
dgs2 = dgs2.set_index('observation_date')['DGS2']
treasury_2y_q = dgs2.resample('QE').mean()
treasury_2y_q.name = 'treasury_2y'
print(f"2Y (FRED): {len(treasury_2y_q)} quarters, "
      f"{treasury_2y_q.index.min().date()} to {treasury_2y_q.index.max().date()}")

# --- Build treasury table ---
treasury_quarterly = treasury_10y_q.to_frame().join(treasury_2y_q, how='outer')
treasury_quarterly['yield_spread'] = treasury_quarterly['treasury_10y'] - treasury_quarterly['treasury_2y']
treasury_quarterly = treasury_quarterly.reset_index()
treasury_quarterly.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
if 'date' not in treasury_quarterly.columns:
    treasury_quarterly = treasury_quarterly.reset_index()
    treasury_quarterly.columns = ['date'] + list(treasury_quarterly.columns[1:])
print(f"Yield spread range: {treasury_quarterly['yield_spread'].min():.2f} to "
      f"{treasury_quarterly['yield_spread'].max():.2f}")

# --- Load panel and merge ---
panel = pd.read_csv('panel_features.csv', parse_dates=['date'])
panel = panel.sort_values(['RegionName', 'date']).reset_index(drop=True)
print(f"\nPanel loaded: {panel.shape}")
panel = panel.merge(treasury_quarterly, on='date', how='left')

# --- Engineer treasury features (Tier 1 only: 10Y) ---
panel['treasury_10y_lag1'] = panel.groupby('RegionName')['treasury_10y'].shift(1)
panel['treasury_10y_lag2'] = panel.groupby('RegionName')['treasury_10y'].shift(2)
panel['treasury_10y_qoq_pct'] = panel.groupby('RegionName')['treasury_10y'].pct_change() * 100
panel['treasury_10y_roll4q'] = panel.groupby('RegionName')['treasury_10y'].transform(
    lambda x: x.rolling(4, min_periods=4).mean()
)

tier1_features = ['treasury_10y', 'treasury_10y_lag1', 'treasury_10y_lag2',
                  'treasury_10y_qoq_pct', 'treasury_10y_roll4q']

# --- Clean: drop rows where tier 1 features are NaN ---
panel = panel.dropna(subset=tier1_features).reset_index(drop=True)
print(f"V2 panel: {panel.shape}, {panel['date'].nunique()} quarters")
print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
print(f"Metros: {panel['RegionName'].nunique()}")
print(f"Target balance: {panel['target'].mean()*100:.1f}% up")

# Save V2 panel locally (will download at end)
panel.to_csv('panel_features_v2.csv', index=False)
print("Saved panel_features_v2.csv")

# --- Define feature set (V2: 25 features) ---
feature_cols = [
    'zhvi', 'cpi', 'mortgage_rate', 'rental_vacancy_rate',
    'zhvi_lag1', 'zhvi_lag2', 'cpi_lag1', 'cpi_lag2',
    'mortgage_rate_lag1', 'mortgage_rate_lag2',
    'rental_vacancy_rate_lag1', 'rental_vacancy_rate_lag2',
    'zhvi_roll2q', 'zhvi_roll4q', 'mortgage_rate_roll2q', 'mortgage_rate_roll4q',
    'cpi_qoq_pct', 'mortgage_rate_qoq_pct', 'zhvi_change_pct_lag1',
    'mortgage_rate_momentum_2q',
    'treasury_10y', 'treasury_10y_lag1', 'treasury_10y_lag2',
    'treasury_10y_qoq_pct', 'treasury_10y_roll4q',
]
print(f"Features: {len(feature_cols)}")
print(f"Null check: {panel[feature_cols].isnull().sum().sum()}")

print("\nStage 1 complete.")


# ============================================================
# STAGE 2: HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: HYPERPARAMETER TUNING")
print("Grid search with walk-forward validation — all 493 metros")
print("=" * 60)

# --- Walk-forward backtest function ---
def walk_forward_backtest(data, feature_cols, rf_params, initial_train_q=40, label=""):
    """Walk-forward backtest. Returns results dict or None."""
    quarters = sorted(data['date'].unique())
    if len(quarters) <= initial_train_q:
        print(f"  [{label}] ERROR: only {len(quarters)} quarters")
        return None

    all_actuals, all_preds, all_quarters = [], [], []
    last_clf = None

    for i in range(initial_train_q, len(quarters)):
        train = data[data['date'].isin(quarters[:i])]
        test = data[data['date'] == quarters[i]]
        if len(test) == 0:
            continue
        clf = RandomForestClassifier(**rf_params, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
        clf.fit(train[feature_cols].values, train['target'].values)
        preds = clf.predict(test[feature_cols].values)
        last_clf = clf
        all_actuals.extend(test['target'].values)
        all_preds.extend(preds)
        all_quarters.extend([quarters[i]] * len(test))

    if last_clf is None:
        return None

    a, p = np.array(all_actuals), np.array(all_preds)
    return {
        'label': label, 'params': rf_params,
        'accuracy': accuracy_score(a, p),
        'precision': precision_score(a, p),
        'recall': recall_score(a, p),
        'f1': f1_score(a, p),
        'confusion_matrix': confusion_matrix(a, p),
        'importances': pd.Series(last_clf.feature_importances_, index=feature_cols).sort_values(ascending=False),
        'results_df': pd.DataFrame({'date': all_quarters, 'actual': a, 'predicted': p}),
    }

# --- Grid definition ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, None],
    'min_samples_split': [10, 30],
    'min_samples_leaf': [1, 5],
}

keys = list(param_grid.keys())
combos = list(product(*[param_grid[k] for k in keys]))
print(f"Combinations: {len(combos)}")
print(f"Parameters: {param_grid}\n")

# --- Run grid search ---
grid_results = []
t0 = time.time()

for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    t_start = time.time()
    result = walk_forward_backtest(panel, feature_cols, params,
                                   label=f"Config {idx+1}/{len(combos)}")
    elapsed = time.time() - t_start
    total_elapsed = time.time() - t0

    if result:
        grid_results.append(result)
        remaining = (total_elapsed / (idx + 1)) * (len(combos) - idx - 1)
        print(f"  [{idx+1:>2}/{len(combos)}] Acc: {result['accuracy']:.4f} | "
              f"F1: {result['f1']:.4f} | {elapsed/60:.1f}min | "
              f"~{remaining/60:.0f}min left | "
              f"n_est={params['n_estimators']} depth={params['max_depth']} "
              f"split={params['min_samples_split']} leaf={params['min_samples_leaf']}")

total_time = time.time() - t0
print(f"\nGrid search complete: {total_time/60:.1f} minutes")

# --- Rank results ---
grid_results.sort(key=lambda x: x['accuracy'], reverse=True)

print(f"\n{'Rank':<5} {'Accuracy':>9} {'F1':>9} {'Prec':>9} {'Rec':>9}  Parameters")
print("-" * 90)
for i, r in enumerate(grid_results[:10]):
    p = r['params']
    print(f"{i+1:<5} {r['accuracy']:>9.4f} {r['f1']:>9.4f} {r['precision']:>9.4f} "
          f"{r['recall']:>9.4f}  n_est={p['n_estimators']} depth={p['max_depth']} "
          f"split={p['min_samples_split']} leaf={p['min_samples_leaf']}")

best = grid_results[0]
worst = grid_results[-1]
print(f"\nBest:  {best['accuracy']:.4f} — {best['params']}")
print(f"Worst: {worst['accuracy']:.4f} — {worst['params']}")
print(f"Spread: {best['accuracy'] - worst['accuracy']:.4f}")

# --- Run V2 baseline (prior config) for comparison ---
print("\nRunning V2 baseline config for comparison...")
v2_params = {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 1}
v2_result = walk_forward_backtest(panel, feature_cols, v2_params, label="V2 Baseline")
if v2_result:
    print(f"  V2 Baseline: Acc {v2_result['accuracy']:.4f} | F1 {v2_result['f1']:.4f}")
    print(f"  Best vs V2 lift: Acc {best['accuracy'] - v2_result['accuracy']:+.4f}")

print("\nStage 2 complete.")


# ============================================================
# STAGE 3: FEATURE SELECTION
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3: FEATURE SELECTION")
print("Using best hyperparameters, prune low-importance features")
print("=" * 60)

best_params = best['params']
importances = best['importances']
print(f"Using params: {best_params}\n")
print("Features ranked by importance:")
for feat, imp in importances.items():
    print(f"  {feat:<30} {imp:.4f}")

# --- Test pruning levels ---
pruning_results = [{
    'n_features': len(feature_cols), 'label': f'All {len(feature_cols)}',
    'features': feature_cols,
    'accuracy': best['accuracy'], 'f1': best['f1'],
    'precision': best['precision'], 'recall': best['recall'],
}]

for n_drop in [3, 5, 8, 10]:
    if n_drop >= len(feature_cols):
        continue
    top_features = list(importances.index[:len(feature_cols) - n_drop])
    dropped = list(importances.index[len(feature_cols) - n_drop:])
    t_start = time.time()
    result = walk_forward_backtest(panel, top_features, best_params,
                                   label=f"Top {len(top_features)}")
    elapsed = time.time() - t_start

    if result:
        pruning_results.append({
            'n_features': len(top_features), 'label': f'Top {len(top_features)}',
            'features': top_features,
            'accuracy': result['accuracy'], 'f1': result['f1'],
            'precision': result['precision'], 'recall': result['recall'],
        })
        print(f"\n  Top {len(top_features)}: Acc {result['accuracy']:.4f} | "
              f"F1 {result['f1']:.4f} | {elapsed/60:.1f}min")
        print(f"  Dropped: {dropped}")

print(f"\n{'Features':>10} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 55)
for pr in pruning_results:
    print(f"{pr['n_features']:>10} {pr['accuracy']:>10.4f} {pr['f1']:>10.4f} "
          f"{pr['precision']:>10.4f} {pr['recall']:>10.4f}")

best_pruning = max(pruning_results, key=lambda x: x['accuracy'])
print(f"\nBest feature count: {best_pruning['n_features']} "
      f"(Acc: {best_pruning['accuracy']:.4f})")

print("\nStage 3 complete.")


# ============================================================
# STAGE 4: FINAL COMPARISON & REPORT
# ============================================================
print("\n" + "=" * 60)
print("STAGE 4: FINAL COMPARISON")
print("=" * 60)

final_features = best_pruning['features']
final_params = best_params
print(f"Final model: {len(final_features)} features")
print(f"Final params: {final_params}")

final_result = walk_forward_backtest(panel, final_features, final_params,
                                     label="Final Tuned+Pruned")

if final_result and v2_result:
    quarters = sorted(panel['date'].unique())
    pred_data = panel[panel['date'].isin(quarters[40:])]
    naive_acc = pred_data['target'].mean()

    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    print(f"{'Naive (always up)':<25} {naive_acc:>10.4f} {naive_acc:>10.4f} {'1.0000':>10} "
          f"{2*naive_acc/(naive_acc+1):>10.4f}")
    print(f"{'V2 Baseline':<25} {v2_result['accuracy']:>10.4f} {v2_result['precision']:>10.4f} "
          f"{v2_result['recall']:>10.4f} {v2_result['f1']:>10.4f}")
    print(f"{'Best Grid Config':<25} {best['accuracy']:>10.4f} {best['precision']:>10.4f} "
          f"{best['recall']:>10.4f} {best['f1']:>10.4f}")
    print(f"{'Final (Tuned+Pruned)':<25} {final_result['accuracy']:>10.4f} "
          f"{final_result['precision']:>10.4f} {final_result['recall']:>10.4f} "
          f"{final_result['f1']:>10.4f}")

    print(f"\nFinal vs V2:    Acc {final_result['accuracy']-v2_result['accuracy']:+.4f}, "
          f"F1 {final_result['f1']-v2_result['f1']:+.4f}")
    print(f"Final vs Naive: Acc {final_result['accuracy']-naive_acc:+.4f}")

    # Period breakdown
    print("\nAccuracy by Period:")
    df = final_result['results_df'].copy()
    df['correct'] = (df['actual'] == df['predicted']).astype(int)
    q_acc = df.groupby('date')['correct'].mean()
    q_acc.index = pd.to_datetime(q_acc.index)
    for label, start, end in [("2007-2011","2007","2012"), ("2012-2016","2012","2017"),
                               ("2017-2021","2017","2022"), ("2022-2025","2022","2026")]:
        mask = (q_acc.index >= start) & (q_acc.index < end)
        if mask.sum() > 0:
            print(f"  {label}: {q_acc[mask].mean():.4f} ({mask.sum()} quarters)")

    # Confusion matrix
    cm = final_result['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Down  Predicted Up")
    print(f"Actual Down      {cm[0,0]:>14,}  {cm[0,1]:>12,}")
    print(f"Actual Up        {cm[1,0]:>14,}  {cm[1,1]:>12,}")

    # Feature importances
    print(f"\nFeature Importances (Final Model):")
    for feat, imp in final_result['importances'].items():
        print(f"  {feat:<30} {imp:.4f}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
accs = [r['accuracy'] for r in grid_results]
f1s = [r['f1'] for r in grid_results]
ax.scatter(accs, f1s, c='#1565C0', alpha=0.7, s=60)
ax.scatter([best['accuracy']], [best['f1']], c='#E53935', s=150, marker='*',
           zorder=5, label='Best config')
if v2_result:
    ax.scatter([v2_result['accuracy']], [v2_result['f1']], c='#4CAF50', s=150,
               marker='D', zorder=5, label='V2 baseline')
ax.set_xlabel('Accuracy'); ax.set_ylabel('F1 Score')
ax.set_title('Grid Search: Accuracy vs F1'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
n_feats = [pr['n_features'] for pr in pruning_results]
p_accs = [pr['accuracy'] for pr in pruning_results]
ax.plot(n_feats, p_accs, 'o-', color='#1565C0', linewidth=2, markersize=8)
ax.set_xlabel('Number of Features'); ax.set_ylabel('Accuracy')
ax.set_title('Feature Pruning: Accuracy vs Count'); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for r, name, color in [(v2_result, 'V2 Baseline', '#90CAF9'),
                         (final_result, 'Final Tuned', '#1565C0')]:
    if r is None: continue
    df = r['results_df'].copy()
    df['correct'] = (df['actual'] == df['predicted']).astype(int)
    q_acc = df.groupby('date')['correct'].mean()
    q_acc.index = pd.to_datetime(q_acc.index)
    roll = q_acc.rolling(4).mean()
    ax.plot(roll.index, roll.values, color=color, linewidth=2, label=name)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
ax.set_title('4Q Rolling Accuracy: V2 vs Tuned'); ax.legend()
ax.set_ylim(0.3, 0.95); ax.grid(True, alpha=0.3)

if final_result:
    ax = axes[1, 1]
    imp = final_result['importances'].sort_values()
    colors = ['#E53935' if 'treasury' in f else '#1565C0' for f in imp.index]
    ax.barh(range(len(imp)), imp.values, color=colors)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=7)
    ax.set_title('Final Feature Importances (red = treasury)'); ax.grid(True, axis='x', alpha=0.3)

plt.suptitle('Hyperparameter Tuning & Feature Selection Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tuning_results.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Save and download files ---
grid_df = pd.DataFrame([{
    'accuracy': r['accuracy'], 'f1': r['f1'],
    'precision': r['precision'], 'recall': r['recall'],
    **r['params']
} for r in grid_results])
grid_df.to_csv('grid_search_results.csv', index=False)

print("\n" + "=" * 60)
print("DOWNLOADING OUTPUT FILES TO YOUR LOCAL MACHINE")
print("=" * 60)
for fname in ['panel_features_v2.csv', 'grid_search_results.csv', 'tuning_results.png']:
    if os.path.exists(fname):
        files.download(fname)
        print(f"  Downloaded: {fname}")

print("\n" + "=" * 60)
print("ITERATION LOG")
print("=" * 60)
print(f"Step: Hyperparameter Tuning + Feature Selection")
print(f"Features: {len(final_features) if final_result else len(feature_cols)}")
if final_result:
    print(f"Final features: {final_features}")
print(f"Best params: {final_params}")
if final_result:
    print(f"Accuracy: {final_result['accuracy']:.4f}")
    print(f"F1: {final_result['f1']:.4f}")
    print(f"Precision: {final_result['precision']:.4f}")
    print(f"Recall: {final_result['recall']:.4f}")
print("\nDone!")

