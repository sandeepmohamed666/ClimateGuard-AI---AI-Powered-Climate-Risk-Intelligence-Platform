import json
from pathlib import Path
p = Path('notebooks/03_feature_engineering.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
old_line = 'engineer.add_statistical_features(compute_quantiles=False, windows=[3,6,12])\n'
new_line = 'engineer.add_statistical_features(windows=[3,6,12], quantiles=[])\n'
replaced = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if old_line in src:
            cell['source'] = [line.replace(old_line, new_line) for line in cell['source']]
            replaced = True
            break
if not replaced:
    raise SystemExit('Did not find the target code line to replace')
p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('Updated notebook cell')
