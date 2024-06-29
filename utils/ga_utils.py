import pandas as pd
from ga.check_monotonicity import check_monotonicity as check_monotonicity_func
import numpy as np
import re

# function to calculate zscore for the data
def zscore_data(df):
    # Create a copy of the dataframe to avoid modifying the original
    zscored_df = df.copy()
    
    # Preserve 'Well Name', 'stage', 'data_id', and 'well_id' columns
    preserved_columns = ['Well Name', 'stage', 'data_id', 'well_id']
    preserved_data = {col: zscored_df[col] for col in preserved_columns if col in zscored_df.columns}
    
    for column in zscored_df.columns:
        if column not in preserved_columns:
            zscored_df[column] = pd.to_numeric(zscored_df[column], errors='coerce')
            col_mean = zscored_df[column].mean()
            col_std = zscored_df[column].std(ddof=1)
            if col_std != 0:  # Avoid division by zero
                zscored_df[column] = (zscored_df[column] - col_mean) / col_std
            else:
                # If standard deviation is zero, set all values to zero
                zscored_df[column] = 0
    
    # Add back preserved columns
    for col, data in preserved_data.items():
        zscored_df[col] = data
    
    return zscored_df

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

# function to calculate predicted productivity for modelling used in ga_main.py
def calculate_predicted_productivity(zscored_df, response_equation):
    # Extract variable names and coefficients from the equation
    equation_parts = re.findall(r'([-+]?\s*\d*\.?\d*)\s*\*?\s*(\w+)(?:\s*\*\s*(\w+))?', response_equation)
    
    predicted = np.zeros(len(zscored_df))
    
    for coef, var1, var2 in equation_parts:
        coef = float(coef) if coef else 1.0
        if var1 == 'Corrected_Prod':
            continue  # Skip the left side of the equation
        elif var2:  # Interaction term
            if var1 in zscored_df.columns and var2 in zscored_df.columns:
                predicted += coef * zscored_df[var1] * zscored_df[var2]
            else:
                print(f"Warning: Columns {var1} or {var2} not found in DataFrame")
        else:  # Single term
            if var1 in zscored_df.columns:
                predicted += coef * zscored_df[var1]
            else:
                print(f"Warning: Column {var1} not found in DataFrame")
    
    # Add intercept
    intercept_match = re.search(r'Corrected_Prod\s*=\s*([-+]?\d*\.?\d*)', response_equation)
    if intercept_match:
        intercept = float(intercept_match.group(1))
        predicted += intercept
    else:
        print("Warning: Intercept not found in equation")
    
    return predicted