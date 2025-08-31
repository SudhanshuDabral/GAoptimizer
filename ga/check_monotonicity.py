import numpy as np
import pandas as pd
import re

def calculate_median(arr, condition_value):
    medians = []
    for i in range(len(arr)):
        current_window = arr[:i+1]
        # Filter out None values and values that don't meet the condition
        filtered_values = [x for x in current_window if x is not None and x > condition_value]
        if filtered_values:
            median_value = np.median(filtered_values)
        else:
            median_value = 0
        medians.append(median_value)
    return medians

def calculate_cumulative_sum(arr):
    # Replace None values with 0 before calculating cumulative sum
    clean_arr = [x if x is not None else 0 for x in arr]
    return np.cumsum(clean_arr).tolist()

def calculate_ratio(numerator, denominator):
    ratios = []
    for num, denom in zip(numerator, denominator):
        # Handle None values and zero denominators
        if num is None or denom is None or denom == 0:
            ratios.append(0)
        else:
            ratios.append(num / denom)
    return ratios

def calculate_stage(df, column, avg, std):
    stage_values = []
    for i in range(len(df)):
        min_value = min(
            df['median_dp_original'][i],
            df['median_dhpm_original'][i],
            df['dhppm_original'][i],
            df['median_slurry_original'][i],
            df['total_slurry_dp_original'][i],
            df['tee_original'][i],
            df['total_dhppm_original'][i],
            df['total_dh_prop_original'][i],
            df['med_energy_dissipated_original'][i],
            df['med_energy_proxy_original'][i],
            df['med_energy_ratio_original'][i],
            df['total_energy_dissipated_original'][i],
            df['total_energy_proxy_original'][i],
            df['total_energy_ratio_original'][i]
        )
        if min_value > 0:
            value = (df[column][i] - avg) / std
        else:
            value = 0
        stage_values.append(value)
    return stage_values

def check_monotonicity(array_data, df_statistics, response_equation):
    import logging
    
    # Create a DataFrame from array_data with correct column names including new energy columns
    expected_columns = ['pmaxmin_win', 'downhole_win_prop', 'slr_win', 'energyproxy', 'energydissipated']
    if isinstance(array_data, pd.DataFrame):
        df = array_data
    else:
        df = pd.DataFrame(array_data, columns=expected_columns)

    # Log input data structure
    print(f"DEBUG: check_monotonicity called with df shape: {df.shape}")
    print(f"DEBUG: check_monotonicity df columns: {list(df.columns)}")
    print(f"DEBUG: df_statistics keys: {list(df_statistics.keys())}")
    
    # Log sample values from energy columns
    for col in ['energyproxy', 'energydissipated']:
        if col in df.columns:
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            print(f"DEBUG: {col} - non_null: {non_null}, null: {null_count}, sample: {df[col].head(3).tolist()}")

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in input data")

    # Calculate all original parameters
    df['median_dp_original'] = calculate_median(df['pmaxmin_win'], 5)
    df['median_dhpm_original'] = calculate_median(df['downhole_win_prop'], 0)
    df['dhppm_original'] = calculate_ratio(df['median_dhpm_original'], df['median_dp_original'])
    df['median_slurry_original'] = calculate_median(df['slr_win'], 5)
    df['total_slurry_dp_original'] = calculate_cumulative_sum(df['slr_win'])
    df['tee_original'] = calculate_cumulative_sum(df['pmaxmin_win'])
    df['total_prop_original'] = calculate_cumulative_sum(df['downhole_win_prop'])
    df['total_dhppm_original'] = calculate_ratio(df['total_prop_original'], df['tee_original'])
    df['total_dh_prop_original'] = calculate_cumulative_sum(df['downhole_win_prop'])
    
    # Calculate new MATLAB energy columns
    # 1. med_energy_dissipated - similar to median_dp but using energydissipated
    df['med_energy_dissipated_original'] = calculate_median(df['energydissipated'], 0)
    print(f"DEBUG: med_energy_dissipated_original sample: {df['med_energy_dissipated_original'].head(3).tolist()}")
    
    # 2. med_energy_proxy - similar to median_dp but using energyproxy  
    df['med_energy_proxy_original'] = calculate_median(df['energyproxy'], 0)
    print(f"DEBUG: med_energy_proxy_original sample: {df['med_energy_proxy_original'].head(3).tolist()}")
    
    # 3. total_energy_dissipated - cumulative energydissipated
    df['total_energy_dissipated_original'] = calculate_cumulative_sum(df['energydissipated'])
    print(f"DEBUG: total_energy_dissipated_original sample: {df['total_energy_dissipated_original'].head(3).tolist()}")
    
    # 4. total_energy_proxy - cumulative energyproxy
    df['total_energy_proxy_original'] = calculate_cumulative_sum(df['energyproxy'])
    print(f"DEBUG: total_energy_proxy_original sample: {df['total_energy_proxy_original'].head(3).tolist()}")
    
    # 5. med_energy_ratio - median energy dissipated / median energy proxy
    df['med_energy_ratio_original'] = calculate_ratio(df['med_energy_dissipated_original'], df['med_energy_proxy_original'])
    print(f"DEBUG: med_energy_ratio_original sample: {df['med_energy_ratio_original'].head(3).tolist()}")
    
    # 6. total_energy_ratio - total energy dissipated / total energy proxy
    df['total_energy_ratio_original'] = calculate_ratio(df['total_energy_dissipated_original'], df['total_energy_proxy_original'])
    print(f"DEBUG: total_energy_ratio_original sample: {df['total_energy_ratio_original'].head(3).tolist()}")

    # Calculate stage values using statistics from df_statistics
    df['tee_stage'] = calculate_stage(df, 'tee_original', df_statistics['tee']['mean'], df_statistics['tee']['std'])
    df['median_dhpm_stage'] = calculate_stage(df, 'median_dhpm_original', df_statistics['median_dhpm']['mean'], df_statistics['median_dhpm']['std'])
    df['median_dp_stage'] = calculate_stage(df, 'median_dp_original', df_statistics['median_dp']['mean'], df_statistics['median_dp']['std'])
    df['downhole_ppm_stage'] = calculate_stage(df, 'dhppm_original', df_statistics['downhole_ppm']['mean'], df_statistics['downhole_ppm']['std'])
    df['total_dhppm_stage'] = calculate_stage(df, 'total_dhppm_original', df_statistics['total_dhppm']['mean'], df_statistics['total_dhppm']['std'])
    df['total_slurry_dp_stage'] = calculate_stage(df, 'total_slurry_dp_original', df_statistics['total_slurry_dp']['mean'], df_statistics['total_slurry_dp']['std'])
    df['median_slurry_stage'] = calculate_stage(df, 'median_slurry_original', df_statistics['median_slurry']['mean'], df_statistics['median_slurry']['std'])
    df['total_dh_prop_stage'] = calculate_stage(df, 'total_dh_prop_original', df_statistics['total_dh_prop']['mean'], df_statistics['total_dh_prop']['std'])
    
    # Calculate stage values for new MATLAB energy columns
    print(f"DEBUG: About to calculate energy stage values using df_statistics")
    for energy_attr in ['med_energy_proxy', 'med_energy_dissipated', 'med_energy_ratio', 'total_energy_proxy', 'total_energy_dissipated', 'total_energy_ratio']:
        if energy_attr in df_statistics:
            print(f"DEBUG: {energy_attr} stats: mean={df_statistics[energy_attr]['mean']:.6f}, std={df_statistics[energy_attr]['std']:.6f}")
        else:
            print(f"DEBUG: {energy_attr} NOT FOUND in df_statistics!")
    
    df['med_energy_proxy_stage'] = calculate_stage(df, 'med_energy_proxy_original', df_statistics['med_energy_proxy']['mean'], df_statistics['med_energy_proxy']['std'])
    print(f"DEBUG: med_energy_proxy_stage sample: {df['med_energy_proxy_stage'].head(3).tolist()}")
    
    df['med_energy_dissipated_stage'] = calculate_stage(df, 'med_energy_dissipated_original', df_statistics['med_energy_dissipated']['mean'], df_statistics['med_energy_dissipated']['std'])
    print(f"DEBUG: med_energy_dissipated_stage sample: {df['med_energy_dissipated_stage'].head(3).tolist()}")
    
    df['med_energy_ratio_stage'] = calculate_stage(df, 'med_energy_ratio_original', df_statistics['med_energy_ratio']['mean'], df_statistics['med_energy_ratio']['std'])
    print(f"DEBUG: med_energy_ratio_stage sample: {df['med_energy_ratio_stage'].head(3).tolist()}")
    
    df['total_energy_proxy_stage'] = calculate_stage(df, 'total_energy_proxy_original', df_statistics['total_energy_proxy']['mean'], df_statistics['total_energy_proxy']['std'])
    print(f"DEBUG: total_energy_proxy_stage sample: {df['total_energy_proxy_stage'].head(3).tolist()}")
    
    df['total_energy_dissipated_stage'] = calculate_stage(df, 'total_energy_dissipated_original', df_statistics['total_energy_dissipated']['mean'], df_statistics['total_energy_dissipated']['std'])
    print(f"DEBUG: total_energy_dissipated_stage sample: {df['total_energy_dissipated_stage'].head(3).tolist()}")
    
    df['total_energy_ratio_stage'] = calculate_stage(df, 'total_energy_ratio_original', df_statistics['total_energy_ratio']['mean'], df_statistics['total_energy_ratio']['std'])
    print(f"DEBUG: total_energy_ratio_stage sample: {df['total_energy_ratio_stage'].head(3).tolist()}")

    # Calculate Productivity - updated to include new MATLAB energy columns
    stage_columns = ['tee_stage', 'median_dhpm_stage', 'median_dp_stage', 'downhole_ppm_stage', 
                     'total_dhppm_stage', 'total_slurry_dp_stage', 'median_slurry_stage', 'total_dh_prop_stage',
                     'med_energy_proxy_stage', 'med_energy_dissipated_stage', 'med_energy_ratio_stage',
                     'total_energy_proxy_stage', 'total_energy_dissipated_stage', 'total_energy_ratio_stage']
    
    # All potential attributes for monotonicity check - updated to include new MATLAB energy columns
    all_potential_attributes = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 
                           'total_dhppm', 'total_slurry_dp', 'median_slurry', 'total_dh_prop',
                           'med_energy_proxy', 'med_energy_dissipated', 'med_energy_ratio',
                           'total_energy_proxy', 'total_energy_dissipated', 'total_energy_ratio']
    
    # Calculate Productivity using the provided equation
    df['Productivity'] = df.apply(lambda row: calculate_productivity(row, stage_columns, response_equation), axis=1)
    
    # Add additional metrics to check monotonicity for all potential attributes
    for attr in all_potential_attributes:
        attr_col = f"{attr}_stage"
        if attr_col in df.columns:
            # Calculate monotonicity metrics
            sorted_df = df.sort_values(attr_col)
            monotonic_increases = 0
            monotonic_decreases = 0
            non_monotonic_count = 0
            
            prev_prod = sorted_df.iloc[0]['Productivity']
            for i in range(1, len(sorted_df)):
                curr_prod = sorted_df.iloc[i]['Productivity']
                if curr_prod > prev_prod:
                    monotonic_increases += 1
                elif curr_prod == prev_prod:
                    # Still monotonic but not strictly increasing
                    monotonic_increases += 1
                else:
                    # Not monotonically increasing
                    non_monotonic_count += 1
                prev_prod = curr_prod
            
            # For hydraulic fracturing attributes, we want INCREASING monotonicity
            monotonic_percent = monotonic_increases / (len(sorted_df) - 1) * 100
            
            # Store monotonicity metrics
            df[f"{attr}_monotonicity_pct"] = monotonic_percent
            df[f"{attr}_monotonicity_dir"] = "increasing" if monotonic_percent >= 75 else "not consistently increasing"
            df[f"{attr}_non_monotonic_pct"] = non_monotonic_count / (len(sorted_df) - 1) * 100
    
    # Export stage columns and parameters to separate sheets
    stage_df = df[stage_columns].copy()
    params_df = df[[
        'median_dp_original', 'median_dhpm_original', 'dhppm_original',
        'median_slurry_original', 'total_slurry_dp_original', 'tee_original',
        'total_prop_original', 'total_dhppm_original', 'total_dh_prop_original',
        'med_energy_proxy_original', 'med_energy_dissipated_original', 'med_energy_ratio_original',
        'total_energy_proxy_original', 'total_energy_dissipated_original', 'total_energy_ratio_original'
    ]].copy()

    return df

def calculate_productivity(row, stage_columns, response_equation):
    # print("Debug: Starting calculate_productivity function")
    if sum(row[col] for col in stage_columns) == 0:
        # print("Debug: Sum of stage columns is zero, returning 0")
        return 0
    else:
        # Modify the response equation to use stage column names
        modified_equation = response_equation.replace("Corrected_Prod", "result")
        # print(f"Debug: Initial modified equation: {modified_equation}")
        
        # Replace feature names with their stage equivalents
        for col in stage_columns:
            original_feature = col.replace('_stage', '')
            modified_equation = re.sub(r'\b' + original_feature + r'\b', col, modified_equation)
        # print(f"Debug: Equation after replacing feature names: {modified_equation}")
        
        # Handle squared terms
        modified_equation = re.sub(r'(\w+)\^2', r'\1 * \1', modified_equation)
        # print(f"Debug: Final modified equation: {modified_equation}")
        
        # Create a dictionary of variables for eval
        variables = row.to_dict()
        variables['result'] = 0
        
        # Evaluate the modified equation
        try:
            exec(modified_equation, {'__builtins__': None}, variables)
            # print(f"Debug: Calculated result: {variables['result']}")
        except Exception as e:
            # print(f"Debug: Error in equation evaluation: {str(e)}")
            return None
        
        return variables['result']
    
# Example usage (to be called from ga_main.py):
# result_df = check_monotonicity(array_data, st.session_state.df_statistics, st.session_state.result[2])