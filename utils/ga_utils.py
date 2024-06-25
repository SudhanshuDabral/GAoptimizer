import pandas as pd
from ga.check_monotonicity import check_monotonicity as check_monotonicity_func

# function to calculate zscore for the data
def zscore_data(df):
    for column in df.columns:
        if column != 'stage':
            df[column] = pd.to_numeric(df[column], errors='coerce')
            col_mean = df[column].mean()
            col_std = df[column].std(ddof=1)
            df[column] = (df[column] - col_mean) / col_std
    return df

# function to calculate statistics for the original data for modelling used in ga_main.py
def calculate_df_statistics(df):
    columns_to_process = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 'total_slurry_dp', 'median_slurry']
    stats = {}
    for col in columns_to_process:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(ddof=1)
            }
    return stats

# function to check monotonicity of the data for modelling used in ga_main.py
def batch_monotonicity_check(stages, well_id, get_array_data_func, df_statistics, equation):
    results = {}
    for stage in stages:
        array_data = get_array_data_func(well_id, stage)
        if array_data is not None:
            if not isinstance(array_data, pd.DataFrame):
                array_data = pd.DataFrame(array_data)
            result_df = check_monotonicity_func(array_data, df_statistics, equation)
            results[stage] = result_df
    return results

# function to validate custom equation for modelling used in ga_main.py
def validate_custom_equation(equation):
    valid_features = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 'total_slurry_dp', 'median_slurry']
    
    # Check if equation starts with "Corrected_Prod ="
    if not equation.strip().startswith("Corrected_Prod ="):
        return False, "Equation must start with 'Corrected_Prod ='"
    
    # Remove "Corrected_Prod =" from the equation for further processing
    equation = equation.replace("Corrected_Prod =", "").strip()
    
    # Check for valid features
    for feature in valid_features:
        equation = equation.replace(feature, "x")
    
    # Replace ^2 with **2 for proper Python syntax
    equation = equation.replace("^2", "**2")
    
    # Remove all spaces
    equation = equation.replace(" ", "")
    
    # Check for valid characters
    valid_chars = set('x+-*/().**0123456789')
    if not all(char in valid_chars for char in equation):
        return False, "Equation contains invalid characters"
    
    # Check for balanced parentheses
    if equation.count('(') != equation.count(')'):
        return False, "Unbalanced parentheses in equation"
    
    # Try to evaluate the equation
    try:
        x = 1  # Dummy value for testing
        eval(equation)
    except Exception as e:
        return False, f"Invalid equation structure: {str(e)}"
    
    return True, "Equation is valid"