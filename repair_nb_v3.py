import json
import os

def fix_notebook():
    file_path = r'd:\2026\python_project_swedish_electricity_forecast\notebooks\05_W7_External_Features_Optimization_and_SHAP.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if not source: continue
            
            # Combine into single string
            if isinstance(source, list):
                content = "".join(source)
            else:
                content = source
            
            # Replace quote + real newline with quote + \n
            # This handles cases like print("\nSomething") where the \n was a real newline
            content = content.replace('"\n', '"\\n')
            content = content.replace("'\n", "'\\n")
            
            # Also handle potentially double escaped ones if they became part of the problem
            # But the primary issue is the quote-newline sequence.
            
            # Re-split into lines
            cell['source'] = [line + '\n' for line in content.splitlines()]
            # Clean up the last newline to not double it if content didn't end with one
            if not content.endswith('\n') and cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook repaired (Fixing split strings).")

if __name__ == "__main__":
    fix_notebook()
