import json
import os
import re

def update_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    changes_made = 0
    fig_counter = 1
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            modified = False
            
            for line in source:
                # Check for plt.show()
                if re.search(r'plt\.show\(\)', line):
                    # Inject savefig before show
                    indent = line[:len(line) - len(line.lstrip())]
                    fig_name = f'outputs/showdown_fig_{fig_counter:02d}.pdf'
                    savefig_line = f"{indent}plt.savefig('../{fig_name}', dpi=300, bbox_inches='tight')\n"
                    new_source.append(savefig_line)
                    fig_counter += 1
                    modified = True
                new_source.append(line)
                
            if modified:
                cell['source'] = new_source
                changes_made += 1
                
    if changes_made > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {notebook_path}. Added {fig_counter - 1} savefig commands.")
    else:
        print(f"No plt.show() found in {notebook_path}.")

if __name__ == '__main__':
    # Update the static split showdown notebook
    target_nb = r'c:\Users\fe_ma\Projects\quant-ai-lab\02_density_forecasting\notebooks\06_static_split_showdown.ipynb'
    if os.path.exists(target_nb):
        update_notebook(target_nb)
    else:
        print(f"Could not find {target_nb}")
