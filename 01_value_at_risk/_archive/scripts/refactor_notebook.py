import json
import os

path = r'C:\Users\fe_ma\AFMA_Repos\quant-ai-lab\01_value_at_risk\notebooks\value_at_risk.ipynb'
out_path = r'C:\Users\fe_ma\AFMA_Repos\quant-ai-lab\01_value_at_risk\notebooks\01_value_at_risk_clean.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. New Cell for GARCH Model
garch_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from value_at_risk.models.garch_model import calculate_garch_var\n",
        "\n",
        "print('Calculating GARCH(1,1) Baseline VaR...')\n",
        "garch_column = '7. GARCH(1,1)'\n",
        "# The function expects unscaled returns, we pass df['log_ret'] which is unscaled\n",
        "garch_var = calculate_garch_var(df['log_ret'], window=ROLLING_WINDOW, alpha=ALPHA)\n",
        "df_results = pd.concat([df_results, pd.Series(garch_var, name=garch_column)], axis=1)\n",
        "print('GARCH calculation complete.')\n"
    ]
}

garch_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 8.3 New Baseline: GARCH(1,1)\n",
        "\n",
        "To make this analysis complete, we introduce a traditional GARCH(1,1) model. \n",
        "This serves as a strong volatility-aware baseline against our Deep Learning approaches."
    ]
}


new_cells = []
for cell in nb.get('cells', []):
    # Clear outputs to make it fresh and clean
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        
        # Patch paths in code
        for i, line in enumerate(cell['source']):
            if '../results' in line:
                cell['source'][i] = line.replace('../results', '../outputs/metrics')
            if '../images' in line:
                cell['source'][i] = line.replace('../images', '../outputs/plots')
                
    elif cell['cell_type'] == 'markdown':
        # Patch paths in markdown
        for i, line in enumerate(cell['source']):
            if '../results' in line:
                cell['source'][i] = line.replace('../results', '../outputs/metrics')
            if '../images' in line:
                cell['source'][i] = line.replace('../images', '../outputs/plots')

    new_cells.append(cell)
    
    # Inject GARCH right after Historical VaR is added to df_results
    if cell['cell_type'] == 'code':
        code_str = "".join(cell['source'])
        if "column = '6. Historical VaR'" in code_str and "pd.concat" in code_str:
            new_cells.append(garch_md_cell)
            new_cells.append(garch_code_cell)


nb['cells'] = new_cells

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Refactored notebook successfully saved to: {out_path}")
