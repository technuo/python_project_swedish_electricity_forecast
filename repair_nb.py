import json
import os

def fix_notebook():
    file_path = r'd:\2026\python_project_swedish_electricity_forecast\notebooks\05_W7_External_Features_Optimization_and_SHAP.ipynb'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if not source:
                continue
            
            new_source = []
            for line in source:
                # 1. Fix backslashes in f-strings: {\' -> {'
                # This fixes the SyntaxError: f-string expression part cannot include a backslash
                fixed_line = line.replace("{\\'", "{'")
                fixed_line = fixed_line.replace("\\'}", "'}")
                fixed_line = fixed_line.replace("[\\'", "['")
                fixed_line = fixed_line.replace("\\']", "']")
                
                # 2. Fix literal newlines in print strings that should be escaped
                # Look for "print(" followed by a newline (which shows up as \n in JSON string)
                # Correct: print("\n...") in Python source should be print("\\n...") in JSON
                # If we see print("\n, we change it to print("\\n
                fixed_line = fixed_line.replace('print("\\n', 'print("\\\\n')
                fixed_line = fixed_line.replace('print(f"\\n', 'print(f"\\\\n')
                
                new_source.append(fixed_line)
            
            cell['source'] = new_source

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook repaired successfully.")

if __name__ == "__main__":
    fix_notebook()
