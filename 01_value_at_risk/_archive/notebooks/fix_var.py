import json
import os

p = r'c:\Users\fe_ma\Projects\quant-ai-lab\01_value_at_risk\notebook\value_at_risk.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix imports and sys.path
for c in nb['cells']:
    if c['cell_type'] == 'code':
        new_source = []
        for line in c['source']:
            # Replace 'from src.' with 'from value_at_risk.'
            if 'from src.' in line:
                line = line.replace('from src.', 'from value_at_risk.')
            # Remove sys.path modifications
            if 'sys.path.append(project_root)' in line:
                continue
            if 'if project_root not in sys.path:' in line:
                continue
            if 'project_root = os.path.abspath(os.path.join(os.getcwd(), \'..\'))' in line:
                continue
            new_source.append(line)
        c['source'] = new_source

# Create new cell to save results to JSON for the Web App
json_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "save_to_json_webapp",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import os\n",
    "\n",
    "results_path = '../outputs/var_results.json'\n",
    "os.makedirs(os.path.dirname(results_path), exist_ok=True)\n",
    "\n",
    "try:\n",
    "    with open(results_path, 'r', encoding='utf-8') as f:\n",
    "        all_results = json.load(f)\n",
    "except (FileNotFoundError, json.JSONDecodeError):\n",
    "    all_results = {}\n",
    "\n",
    "# Convert summary table to dictionary\n",
    "dict_results = summary_table.to_dict(orient='index')\n",
    "all_results[TICKER] = dict_results\n",
    "\n",
    "with open(results_path, 'w', encoding='utf-8') as f:\n",
    "    json.dump(all_results, f, indent=4)\n",
    "\n",
    "print(f\"Saved results for {TICKER} to {results_path}\")"
   ]
}

nb['cells'].append(json_cell)

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated successfully.")
