import json

p = r'c:\Users\fe_ma\Projects\quant-ai-lab\01_value_at_risk\notebook\value_at_risk.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)
    
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

with open('dump.txt', 'w', encoding='utf-8') as f:
    f.write('\n---CELL---\n'.join([''.join(c['source']) for c in code_cells[-10:]]))
