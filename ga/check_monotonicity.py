import numpy as np
import pandas as pd
import re

def calculate_median(arr, condition_value):
    medians = []
    for i in range(len(arr)):
        current_window = arr[:i+1]
        filtered_values = [x for x in current_window if x > condition_value]
        if filtered_values:
            median_value = np.median(filtered_values)
        else:
            median_value = 0
        medians.append(median_value)
    return medians

def calculate_cumulative_sum(arr):
    return np.cumsum(arr).tolist()

def calculate_ratio(numerator, denominator):
    ratios = []
    for num, denom in zip(numerator, denominator):
        if denom == 0:
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
            df['total_dhppm_original'][i]
        )
        if min_value > 0:
            value = (df[column][i] - avg) / std
        else:
            value = 0
        stage_values.append(value)
    return stage_values

def check_monotonicity(array_data, df_statistics, response_equation):
    # Create a DataFrame from array_data with correct column names
    expected_columns = ['pmaxmin_win', 'downhole_win_prop', 'slr_win']
    if isinstance(array_data, pd.DataFrame):
        df = array_data
    else:
        df = pd.DataFrame(array_data, columns=expected_columns)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in input data")

    # Calculate all parameters
    df['median_dp_original'] = calculate_median(df['pmaxmin_win'], 5)
    df['median_dhpm_original'] = calculate_median(df['downhole_win_prop'], 0)
    df['dhppm_original'] = calculate_ratio(df['median_dhpm_original'], df['median_dp_original'])
    df['median_slurry_original'] = calculate_median(df['slr_win'], 5)
    df['total_slurry_dp_original'] = calculate_cumulative_sum(df['slr_win'])
    df['tee_original'] = calculate_cumulative_sum(df['pmaxmin_win'])
    df['total_prop_original'] = calculate_cumulative_sum(df['downhole_win_prop'])
    df['total_dhppm_original'] = calculate_ratio(df['total_prop_original'], df['tee_original'])
    
    # Calculate stage values using statistics from df_statistics
    df['tee_stage'] = calculate_stage(df, 'tee_original', df_statistics['tee']['mean'], df_statistics['tee']['std'])
    df['median_dhpm_stage'] = calculate_stage(df, 'median_dhpm_original', df_statistics['median_dhpm']['mean'], df_statistics['median_dhpm']['std'])
    df['median_dp_stage'] = calculate_stage(df, 'median_dp_original', df_statistics['median_dp']['mean'], df_statistics['median_dp']['std'])
    df['downhole_ppm_stage'] = calculate_stage(df, 'dhppm_original', df_statistics['downhole_ppm']['mean'], df_statistics['downhole_ppm']['std'])
    df['total_dhppm_stage'] = calculate_stage(df, 'total_dhppm_original', df_statistics['total_dhppm']['mean'], df_statistics['total_dhppm']['std'])
    df['total_slurry_dp_stage'] = calculate_stage(df, 'total_slurry_dp_original', df_statistics['total_slurry_dp']['mean'], df_statistics['total_slurry_dp']['std'])
    df['median_slurry_stage'] = calculate_stage(df, 'median_slurry_original', df_statistics['median_slurry']['mean'], df_statistics['median_slurry']['std'])

    # Calculate Productivity
    stage_columns = ['tee_stage', 'median_dhpm_stage', 'median_dp_stage', 'downhole_ppm_stage', 
                     'total_dhppm_stage', 'total_slurry_dp_stage', 'median_slurry_stage']
    
    df['Productivity'] = df.apply(lambda row: calculate_productivity(row, stage_columns, response_equation), axis=1)

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