import json, pathlib
p=pathlib.Path('notebooks/03_feature_engineering.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    src=''.join(c.get('source',[]))
    if 'WeatherFeatureEngineer' in src or 'weather_features' in src or 'from src.feature_engineering' in src or 'import src' in src:
        print('CELL',i,'TYPE',c['cell_type'])
        print('\n'.join(src.splitlines()[:20]))
        print('---')
