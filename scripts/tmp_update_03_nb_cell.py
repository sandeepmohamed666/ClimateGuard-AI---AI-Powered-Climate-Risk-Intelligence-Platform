import json
from pathlib import Path
p = Path('notebooks/03_feature_engineering.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
# find the markdown cell header '## 16. Add Advanced Engineered Features' then replace the next code cell
for i, c in enumerate(nb['cells']):
    if c['cell_type']=='markdown' and any('## 16. Add Advanced Engineered Features' in s for s in c['source']):
        # find next code cell index
        j = i+1
        while j < len(nb['cells']) and nb['cells'][j]['cell_type']!='code':
            j += 1
        if j < len(nb['cells']):
            new_source = [
                "import time\n",
                "# Optimized single-run advanced features (no duplicates)\n",
                "start = time.time()\n",
                "engineer.add_statistical_features(compute_quantiles=False, windows=[3,6,12])\n",
                "print(f\"Statistical features done: {time.time()-start:.2f}s\")\n",
                "start = time.time()\n",
                "engineer.add_expanding_features()\n",
                "print(f\"Expanding features done: {time.time()-start:.2f}s\")\n",
                "start = time.time()\n",
                "engineer.add_exponential_features(spans=[3,6])\n",
                "print(f\"Exponential features done: {time.time()-start:.2f}s\")\n",
                "start = time.time()\n",
                "engineer.add_lag_features(lags=[1,3,6])\n",
                "print(f\"Lag features done: {time.time()-start:.2f}s\")\n",
                "start = time.time()\n",
                "engineer.add_trend_features(windows=[3,6])\n",
                "print(f\"Trend features done: {time.time()-start:.2f}s\")\n",
                "start = time.time()\n",
                "engineer.add_interaction_features()\n",
                "print(f\"Interaction features done: {time.time()-start:.2f}s\")\n",
                "start = time.time()\n",
                "engineer.add_derived_weather_features()\n",
                "print(f\"Derived weather features done: {time.time()-start:.2f}s\")\n",
                "target_col = \"precipitation\" if \"precipitation\" in engineer.df.columns else None\n",
                "start = time.time()\n",
                "engineer.prepare_forecasting_features(target_col=target_col, future_horizon=3, lags=[1,3], windows=[3], spans=[3])\n",
                "print(f\"Forecasting features done: {time.time()-start:.2f}s\")\n",
                "print(\"Added advanced engineered features (optimized single run).\")\n",
                "print(f\"Forecast target column: {target_col}\")\n",
            ]
            nb['cells'][j]['source'] = new_source
            p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
            print('Replaced cell at index', j)
            break
print('Done')
