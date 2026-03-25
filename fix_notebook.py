import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('c:/Users/abfat/Desktop/HX3/HX3_Weight_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── CELL 12 FIX ──────────────────────────────────────────────────────────────
src12 = ''.join(cells[12]['source'])

old_sweep = (
    "        if feature == 'frontal_area':\n"
    "            feat_vals_dataset.append(derived['frontal_area'])\n"
    "            # All parameters that contribute to frontal area\n"
    "            sweep_indices = [idx_overall_width, idx_coolant_diam, idx_channel_height, idx_num_air_layers]\n"
    "        elif feature == 'as_hyd_diam':\n"
    "            feat_vals_dataset.append(derived['AS_hyd_diam'])\n"
    "            # All geometric parameters (through porosity and A_HT)\n"
    "            sweep_indices = [idx_overall_length, idx_overall_width, idx_channel_height, idx_num_air_layers, idx_strut_diameter, idx_strut_l_d_ratio]\n"
    "        elif feature == 'porosity':\n"
    "            feat_vals_dataset.append(derived['porosity'])\n"
    "            # All geometric parameters affect porosity\n"
    "            sweep_indices = [idx_overall_length, idx_overall_width, idx_channel_height, idx_num_air_layers, idx_strut_diameter, idx_strut_l_d_ratio]"
)

new_sweep = (
    "        if feature == 'frontal_area':\n"
    "            feat_vals_dataset.append(derived['frontal_area'])\n"
    "            # Only HX_overall_width is changed by precise_feature_scaling (analytical solve)\n"
    "            sweep_indices = [idx_overall_width]\n"
    "        elif feature == 'as_hyd_diam':\n"
    "            feat_vals_dataset.append(derived['AS_hyd_diam'])\n"
    "            # Channel dims cancel in the formula; only Strut_Diameter is changed (L_D fixed)\n"
    "            sweep_indices = [idx_strut_diameter]\n"
    "        elif feature == 'porosity':\n"
    "            feat_vals_dataset.append(derived['porosity'])\n"
    "            # Porosity depends ONLY on Strut_l_d_ratio (channel dims cancel out)\n"
    "            sweep_indices = [idx_strut_l_d_ratio]"
)

assert old_sweep in src12, "sweep_indices block not found"
src12 = src12.replace(old_sweep, new_sweep, 1)
print("sweep_indices fixed")

# ── Fix precise_feature_scaling body ─────────────────────────────────────────
old_fn_start = '        """Precise feature scaling using direct mathematical relationships."""\n        x_new = x_ref.copy()\n        \n        # Use the proven simultaneous sweeping approach from drag notebook'
assert old_fn_start in src12, "function start not found"

fn_start_idx = src12.find(old_fn_start)
fn_end_marker = "\n        return x_new"
fn_end_idx = src12.find(fn_end_marker, fn_start_idx) + len(fn_end_marker)

new_fn_body = (
    '        """\n'
    '        Invert derived feature formula analytically / via binary search,\n'
    '        touching the MINIMUM set of inputs.\n'
    '\n'
    '        Simplified maths (channel dims cancel for porosity and AS_hyd_diam):\n'
    '          porosity    = 1 - 0.75pi*(L_D-0.583)/L_D^3  -> depends ONLY on Strut_l_d_ratio\n'
    '          AS_hyd_diam = 4*phi*L_D^3*D/[3pi*(L_D+2/pi-1.5)] -> depends on D and L_D only\n'
    '          frontal_area = width * hx_height             -> solve for HX_overall_width\n'
    '        """\n'
    '        x_new = x_ref.copy()\n'
    '        p5_LD, p95_LD = np.percentile(X_data[:, idx_strut_l_d_ratio], [5, 95])\n'
    '        p5_D,  p95_D  = np.percentile(X_data[:, idx_strut_diameter],  [5, 95])\n'
    '        p5_W,  p95_W  = np.percentile(X_data[:, idx_overall_width],   [5, 95])\n'
    '\n'
    "        if feature_name == 'porosity':\n"
    '            # Binary search on Strut_l_d_ratio (porosity is monotone in L_D)\n'
    '            lo, hi = p5_LD, p95_LD\n'
    '            for _ in range(80):\n'
    '                mid = 0.5 * (lo + hi)\n'
    '                x_t = x_new.copy(); x_t[idx_strut_l_d_ratio] = mid\n'
    "                val = calc_derived_features(x_t)['porosity']\n"
    '                if abs(val - target_feat_value) < 1e-10:\n'
    '                    break\n'
    '                x_t_lo = x_new.copy(); x_t_lo[idx_strut_l_d_ratio] = lo\n'
    "                val_lo = calc_derived_features(x_t_lo)['porosity']\n"
    '                if val_lo < target_feat_value:\n'
    '                    if val < target_feat_value:\n'
    '                        lo = mid\n'
    '                    else:\n'
    '                        hi = mid\n'
    '                else:\n'
    '                    if val > target_feat_value:\n'
    '                        lo = mid\n'
    '                    else:\n'
    '                        hi = mid\n'
    '            x_new[idx_strut_l_d_ratio] = np.clip(0.5 * (lo + hi), p5_LD, p95_LD)\n'
    '\n'
    "        elif feature_name == 'as_hyd_diam':\n"
    '            # AS_hyd_diam = 4*phi(L_D)*L_D^3*D / [3pi*(L_D+2/pi-1.5)]\n'
    '            # Given current L_D (unchanged), solve analytically for Strut_Diameter\n'
    '            L_D = x_new[idx_strut_l_d_ratio]\n'
    "            porosity = calc_derived_features(x_new)['porosity']\n"
    '            denom = 4.0 * porosity * L_D**3\n'
    '            numer = target_feat_value * 3.0 * np.pi * (L_D + 2.0 / np.pi - 1.5)\n'
    '            if denom > 1e-14:\n'
    '                x_new[idx_strut_diameter] = np.clip(numer / denom, p5_D, p95_D)\n'
    '\n'
    "        elif feature_name == 'frontal_area':\n"
    '            # frontal_area = HX_overall_width * hx_overall_height\n'
    '            # hx_height is fixed; solve analytically for HX_overall_width\n'
    '            hx_height = (\n'
    '                (x_new[idx_coolant_diam] + 0.003) * (x_new[idx_num_air_layers] - 1)\n'
    '                + x_new[idx_channel_height] * x_new[idx_num_air_layers]\n'
    '            )\n'
    '            if hx_height > 1e-12:\n'
    '                x_new[idx_overall_width] = np.clip(target_feat_value / hx_height, p5_W, p95_W)\n'
    '\n'
    "        key_map = {'as_hyd_diam': 'AS_hyd_diam', 'frontal_area': 'frontal_area', 'porosity': 'porosity'}\n"
    '        key = key_map.get(feature_name, feature_name)\n'
    "        final_val = calc_derived_features(x_new).get(key, float('nan'))\n"
    '        print(f"      {feature_name}: target={target_feat_value:.6g}  achieved={final_val:.6g}  "\n'
    '              f"err={abs(final_val - target_feat_value):.3g}")\n'
    '        return x_new'
)

src12 = src12[:fn_start_idx] + new_fn_body + src12[fn_end_idx:]
cells[12]['source'] = src12.splitlines(keepends=True)
print("Cell 12 precise_feature_scaling patched OK")

# ── CELL 17 FIX ──────────────────────────────────────────────────────────────
src17 = ''.join(cells[16]['source'])  # cell 16 = ROBUST DERIVED FEATURE SWEEPING

helper = (
    'def _set_derived_feature(x, feature_name, target_value, name_to_idx, X_data):\n'
    '    """\n'
    '    Invert a derived feature formula analytically / via binary search,\n'
    '    touching the MINIMUM set of inputs.\n'
    '\n'
    '    Derived feature math (channel dims cancel in porosity and AS_hyd_diam):\n'
    '      porosity    = 1 - 0.75pi*(L_D-0.583)/L_D^3    depends ONLY on Strut_l_d_ratio\n'
    '      AS_hyd_diam = 4*phi*L_D^3*D/[3pi*(L_D+2/pi-1.5)]  depends on D and L_D only\n'
    '      frontal_area = width * hx_height               solve for HX_overall_width\n'
    '    """\n'
    '    x_new = x.copy()\n'
    "    idx_LD  = name_to_idx['Strut length to diameter ratio']\n"
    "    idx_D   = name_to_idx['Strut Diameter (m)']\n"
    "    idx_W   = name_to_idx['HX overall width (m)']\n"
    "    idx_cd  = name_to_idx['coolant channel diameter (m)']\n"
    "    idx_nal = name_to_idx['Number of air layers/channels']\n"
    "    idx_ch  = name_to_idx['Channel height (m)']\n"
    '\n'
    '    p5_LD, p95_LD = np.percentile(X_data[:, idx_LD], [5, 95])\n'
    '    p5_D,  p95_D  = np.percentile(X_data[:, idx_D],  [5, 95])\n'
    '    p5_W,  p95_W  = np.percentile(X_data[:, idx_W],  [5, 95])\n'
    '\n'
    "    if feature_name == 'porosity':\n"
    '        lo, hi = p5_LD, p95_LD\n'
    '        for _ in range(80):\n'
    '            mid = 0.5 * (lo + hi)\n'
    '            x_t = x_new.copy(); x_t[idx_LD] = mid\n'
    "            val = calc_derived_features(x_t)['porosity']\n"
    '            if abs(val - target_value) < 1e-10:\n'
    '                break\n'
    '            x_t_lo = x_new.copy(); x_t_lo[idx_LD] = lo\n'
    "            val_lo = calc_derived_features(x_t_lo)['porosity']\n"
    '            if val_lo < target_value:\n'
    '                if val < target_value:\n'
    '                    lo = mid\n'
    '                else:\n'
    '                    hi = mid\n'
    '            else:\n'
    '                if val > target_value:\n'
    '                    lo = mid\n'
    '                else:\n'
    '                    hi = mid\n'
    '        x_new[idx_LD] = np.clip(0.5 * (lo + hi), p5_LD, p95_LD)\n'
    '\n'
    "    elif feature_name in ('AS_hyd_diam', 'as_hyd_diam'):\n"
    '        L_D = x_new[idx_LD]\n'
    "        porosity = calc_derived_features(x_new)['porosity']\n"
    '        denom = 4.0 * porosity * L_D**3\n'
    '        numer = target_value * 3.0 * np.pi * (L_D + 2.0 / np.pi - 1.5)\n'
    '        if denom > 1e-14:\n'
    '            x_new[idx_D] = np.clip(numer / denom, p5_D, p95_D)\n'
    '\n'
    "    elif feature_name == 'frontal_area':\n"
    '        hx_height = (\n'
    '            (x_new[idx_cd] + 0.003) * (x_new[idx_nal] - 1)\n'
    '            + x_new[idx_ch] * x_new[idx_nal]\n'
    '        )\n'
    '        if hx_height > 1e-12:\n'
    '            x_new[idx_W] = np.clip(target_value / hx_height, p5_W, p95_W)\n'
    '\n'
    '    return x_new\n'
    '\n'
    '\n'
)

old_sdr_start = 'def sweep_derived_feature_robustly(strategy, n_points, fixed_inputs=None):'
sdr_idx = src17.find(old_sdr_start)
assert sdr_idx >= 0, "sweep_derived_feature_robustly not found"

sdr_end_marker = '\n\ndef create_robust_2d_heatmap'
sdr_end_idx = src17.find(sdr_end_marker, sdr_idx)
assert sdr_end_idx >= 0, "end of sweep_derived_feature_robustly not found"

new_sdr = (
    'def sweep_derived_feature_robustly(strategy, n_points, fixed_inputs=None):\n'
    '    """\n'
    '    Sweep a derived feature using exact analytical/binary-search inversion\n'
    '    of the underlying physics formulas (minimal input modification).\n'
    '    """\n'
    '    target_min, target_max = strategy[\'target_range\']\n'
    '    baseline_inputs = strategy[\'baseline_inputs\'].copy()\n'
    '    if fixed_inputs:\n'
    '        for idx, value in fixed_inputs.items():\n'
    '            baseline_inputs[idx] = value\n'
    '\n'
    '    target_values = np.linspace(target_min, target_max, n_points)\n'
    '    input_names_local = get_input_names()\n'
    '    name_to_idx_local = {n: i for i, n in enumerate(input_names_local)}\n'
    '\n'
    '    input_vectors = []\n'
    '    derived_values = []\n'
    '    for target_val in target_values:\n'
    '        new_inputs = baseline_inputs.copy()\n'
    '        if fixed_inputs:\n'
    '            for idx, value in fixed_inputs.items():\n'
    '                new_inputs[idx] = value\n'
    '        new_inputs = _set_derived_feature(\n'
    '            new_inputs, strategy[\'target_feature\'], target_val, name_to_idx_local, X_data)\n'
    '        if fixed_inputs:\n'
    '            for idx, value in fixed_inputs.items():\n'
    '                new_inputs[idx] = value\n'
    "        key_map = {'AS_hyd_diam': 'AS_hyd_diam', 'as_hyd_diam': 'AS_hyd_diam',\n"
    "                   'frontal_area': 'frontal_area', 'porosity': 'porosity'}\n"
    "        key = key_map.get(strategy['target_feature'], strategy['target_feature'])\n"
    '        actual_derived = calc_derived_features(new_inputs)[key]\n'
    '        input_vectors.append(new_inputs)\n'
    '        derived_values.append(actual_derived)\n'
    '\n'
    '    return np.array(input_vectors), np.array(derived_values)'
)

src17 = src17[:sdr_idx] + helper + new_sdr + src17[sdr_end_idx:]
print("Cell 16 sweep_derived_feature_robustly patched OK")

# ── Fix grid loop in create_robust_2d_heatmap (cell 16) ─────────────────────
probe = "scale_factor = target_val / calc_derived_features(baseline_inputs)[feature_1_name]"
probe_idx = src17.find(probe)
assert probe_idx >= 0, "scale_factor probe not found in cell 16"

block_start_marker = (
    "                # Handle feature 1\n"
    "                if feature_1_name in strategies:\n"
    "                    # Derived feature - use robust sweeping"
)
block_start = src17.find(block_start_marker)
assert block_start >= 0, "grid block start not found"

block_end_marker = "                        baseline_inputs[name_to_idx[feature_2_name]] = F2[i, j]"
block_end = src17.find(block_end_marker, block_start) + len(block_end_marker)
assert block_end > block_start, "grid block end not found"

new_grid = (
    "                # Set direct inputs (no dependencies)\n"
    "                if feature_1_name not in strategies:\n"
    "                    if feature_1_name in name_to_idx:\n"
    "                        baseline_inputs[name_to_idx[feature_1_name]] = F1[i, j]\n"
    "                if feature_2_name not in strategies:\n"
    "                    if feature_2_name in name_to_idx:\n"
    "                        baseline_inputs[name_to_idx[feature_2_name]] = F2[i, j]\n"
    "\n"
    "                # Set derived features in dependency order:\n"
    "                # porosity (sets L_D) must come before AS_hyd_diam (uses L_D to solve for D)\n"
    "                _DERIVED_ORDER = ['porosity', 'AS_hyd_diam', 'frontal_area']\n"
    "                _derived_targets = []\n"
    "                if feature_1_name in strategies:\n"
    "                    _derived_targets.append((feature_1_name, F1[i, j]))\n"
    "                if feature_2_name in strategies:\n"
    "                    _derived_targets.append((feature_2_name, F2[i, j]))\n"
    "                _derived_targets.sort(\n"
    "                    key=lambda x: _DERIVED_ORDER.index(x[0]) if x[0] in _DERIVED_ORDER else 99)\n"
    "                for _feat_name, _target_val in _derived_targets:\n"
    "                    baseline_inputs = _set_derived_feature(\n"
    "                        baseline_inputs, _feat_name, _target_val, name_to_idx, X_data)"
)

src17 = src17[:block_start] + new_grid + src17[block_end:]
cells[16]['source'] = src17.splitlines(keepends=True)
print("Cell 16 grid loop patched OK")

# ── Fix create_robust_2d_heatmapHL grid loop (cell 18) ───────────────────────
src18 = ''.join(cells[18]['source'])

probe18 = "scale_factor = target_val / calc_derived_features(baseline_inputs)[feature_1_name]"
assert probe18 in src18, "scale_factor probe not found in cell 18"

block18_start_marker = (
    "                # Handle feature 1\n"
    "                if feature_1_name in strategies:\n"
    "                    # Derived feature - use robust sweeping"
)
block18_start = src18.find(block18_start_marker)
assert block18_start >= 0, "cell18 grid block start not found"

block18_end_marker = "                        baseline_inputs[name_to_idx[feature_2_name]] = F2[i, j]"
block18_end = src18.find(block18_end_marker, block18_start) + len(block18_end_marker)
assert block18_end > block18_start, "cell18 grid block end not found"

src18 = src18[:block18_start] + new_grid + src18[block18_end:]
cells[18]['source'] = src18.splitlines(keepends=True)
print("Cell 18 grid loop patched OK")

# ── Write back ────────────────────────────────────────────────────────────────
with open('c:/Users/abfat/Desktop/HX3/HX3_Weight_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook saved.")
