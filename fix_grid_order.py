"""Fix grid loop ordering in cells 16 and 18: porosity must be set before AS_hyd_diam."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('c:/Users/abfat/Desktop/HX3/HX3_Weight_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# The current grid block content (same in both cells 16 and 18 after the previous fix)
old_grid = (
    "                # Handle feature 1\n"
    "                if feature_1_name in strategies:\n"
    "                    baseline_inputs = _set_derived_feature(\n"
    "                        baseline_inputs, feature_1_name, F1[i, j], name_to_idx, X_data)\n"
    "                else:\n"
    "                    if feature_1_name in name_to_idx:\n"
    "                        baseline_inputs[name_to_idx[feature_1_name]] = F1[i, j]\n"
    "\n"
    "                # Handle feature 2 (after feature 1 adjustments)\n"
    "                if feature_2_name in strategies:\n"
    "                    baseline_inputs = _set_derived_feature(\n"
    "                        baseline_inputs, feature_2_name, F2[i, j], name_to_idx, X_data)\n"
    "                else:\n"
    "                    if feature_2_name in name_to_idx:\n"
    "                        baseline_inputs[name_to_idx[feature_2_name]] = F2[i, j]"
)

# Replacement: set direct inputs first (no dependencies), then derived features in
# dependency order so porosity (sets L_D) is always applied before AS_hyd_diam
# (which uses the current L_D to analytically solve for Strut_Diameter).
new_grid = (
    "                # Set direct inputs first (no inter-feature dependencies)\n"
    "                if feature_1_name not in strategies:\n"
    "                    if feature_1_name in name_to_idx:\n"
    "                        baseline_inputs[name_to_idx[feature_1_name]] = F1[i, j]\n"
    "                if feature_2_name not in strategies:\n"
    "                    if feature_2_name in name_to_idx:\n"
    "                        baseline_inputs[name_to_idx[feature_2_name]] = F2[i, j]\n"
    "\n"
    "                # Apply derived features in dependency order:\n"
    "                #   porosity  -> sets Strut_l_d_ratio (L_D)\n"
    "                #   AS_hyd_diam -> sets Strut_Diameter using the L_D just fixed above\n"
    "                #   frontal_area -> sets HX_overall_width (independent)\n"
    "                _DERIVED_ORDER = ['porosity', 'AS_hyd_diam', 'frontal_area']\n"
    "                _derived_targets = []\n"
    "                if feature_1_name in strategies:\n"
    "                    _derived_targets.append((feature_1_name, F1[i, j]))\n"
    "                if feature_2_name in strategies:\n"
    "                    _derived_targets.append((feature_2_name, F2[i, j]))\n"
    "                _derived_targets.sort(\n"
    "                    key=lambda x: _DERIVED_ORDER.index(x[0])\n"
    "                    if x[0] in _DERIVED_ORDER else 99)\n"
    "                for _feat, _tgt in _derived_targets:\n"
    "                    baseline_inputs = _set_derived_feature(\n"
    "                        baseline_inputs, _feat, _tgt, name_to_idx, X_data)"
)

patched = 0
for cell_idx in [16, 18]:
    src = ''.join(cells[cell_idx]['source'])
    assert old_grid in src, f"Old grid block not found in cell {cell_idx}"
    src = src.replace(old_grid, new_grid, 1)
    cells[cell_idx]['source'] = src.splitlines(keepends=True)
    print(f"Cell {cell_idx} grid loop ordering fixed")
    patched += 1

print(f"Patched {patched} cells")

with open('c:/Users/abfat/Desktop/HX3/HX3_Weight_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook saved.")
