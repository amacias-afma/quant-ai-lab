import json
import os

p = r'c:\Users\fe_ma\Projects\quant-ai-lab\01_value_at_risk\notebook\value_at_risk.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if already patched
already_patched = any('save_timeseries_webapp' in cell.get('id', '') for cell in nb['cells'])
if already_patched:
    print("Notebook already patched.")
else:
    # Find the cell that exports the summary table
    export_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'summary_table.to_dict()' in ''.join(cell.get('source', [])):
            export_cell_idx = i
            break

    ts_cell = {
       "cell_type": "code",
       "execution_count": None,
       "id": "save_timeseries_webapp",
       "metadata": {},
       "outputs": [],
       "source": [
        "import json\n",
        "import os\n",
        "import numpy as np\n",
        "\n",
        "ts_path = '../outputs/var_timeseries.json'\n",
        "os.makedirs(os.path.dirname(ts_path), exist_ok=True)\n",
        "\n",
        "try:\n",
        "    with open(ts_path, 'r', encoding='utf-8') as f:\n",
        "        all_ts = json.load(f)\n",
        "except (FileNotFoundError, json.JSONDecodeError):\n",
        "    all_ts = {}\n",
        "\n",
        "# Convert df_results to dictionary\n",
        "df_ts = df_results.copy()\n",
        "df_ts.index = df_ts.index.astype(str)\n",
        "\n",
        "# Replace NaN with None for JSON serialization\n",
        "df_ts = df_ts.replace({np.nan: None})\n",
        "\n",
        "all_ts[TICKER] = df_ts.to_dict(orient='list')\n",
        "all_ts[TICKER]['date'] = df_ts.index.tolist()\n",
        "\n",
        "with open(ts_path, 'w', encoding='utf-8') as f:\n",
        "    json.dump(all_ts, f, indent=4)\n",
        "\n",
        "print(f\"Saved timeseries for {TICKER} to {ts_path}\")"
       ]
    }

    if export_cell_idx != -1:
        nb['cells'].insert(export_cell_idx + 1, ts_cell)
    else:
        nb['cells'].append(ts_cell)

    with open(p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook updated successfully with timeseries export.")
