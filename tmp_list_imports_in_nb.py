import json, pathlib
p=pathlib.Path('notebooks/03_feature_engineering.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    src=''.join(c.get('source',[]))
    if 'import ' in src or 'from ' in src:
        print('CELL',i,'TYPE',c['cell_type'])
        for line in src.splitlines()[:20]:
            if 'import ' in line or 'from ' in line:
                print('  ', line)
        print('---')
