import json
import os
import re

def fix_notebook():
    file_path = r'd:\2026\python_project_swedish_electricity_forecast\notebooks\05_W7_External_Features_Optimization_and_SHAP.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if not source: continue
            
            # If source is a string, make it a list for easier processing
            is_single_string = isinstance(source, str)
            if is_single_string:
                content = source
            else:
                content = "".join(source)
            
            # 1. Fix the f-string backslash error
            # Replace any backslash inside curly braces
            # We use a regex to find { ... } and remove backslashes from it
            def remove_bs_in_braces(match):
                return '{' + match.group(1).replace('\\', '') + '}'
            
            content = re.sub(r'\{(.*?)\}', remove_bs_in_braces, content)
            
            # 2. Fix the unterminated string literal error
            # Replace print("\n with print("\\n
            # Also handle f"\n
            content = content.replace('print("\\n', 'print("\\\\n')
            content = content.replace('print(f"\\n', 'print(f"\\\\n')
            content = content.replace('print(\'\\n', 'print(\'\\\\n')
            
            # 3. Specifically fix the "Completed Tasks" etc block which has unescaped \n
            # These often look like print("\nSomething") in the code but print("\nSomething") in JSON
            # which results in a literal newline. 
            # We already did some of this in step 2, but let's be thorough.
            # If there's a " followed by a real newline inside a print statement, it's a problem.
            # But the 'content' here already has real newlines.
            
            if is_single_string:
                cell['source'] = content
            else:
                # Split back into lines for typical notebook format
                lines = content.splitlines(keepends=True)
                cell['source'] = lines

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook repaired with aggressive regex.")

if __name__ == "__main__":
    fix_notebook()
