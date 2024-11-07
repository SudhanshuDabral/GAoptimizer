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
    columns_to_process = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 
                         'total_slurry_dp', 'median_slurry', 'effective_tee', 'effective_mediandp']
    stats = {}
    for col in columns_to_process:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(ddof=1),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
            }
    return stats

def calculate_zscoredf_statistics(df):
    columns_to_process = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 
                         'total_slurry_dp', 'median_slurry', 'effective_tee', 'effective_mediandp']
    stats = {}
    for col in columns_to_process:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(ddof=1),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
            }
    return stats

# function to calculate model sensitivity for the data for modelling used in ga_main.py
def calculate_productivity(values, response_equation):
    try:
        # Modify the response equation to use 'result' instead of 'Computed_Productivity' or 'Corrected_Prod'
        modified_equation = response_equation.replace("Computed_Productivity", "result").replace("Corrected_Prod", "result")
        
        # Extract the right side of the equation
        right_side = modified_equation.split('=')[1].strip()
        
        # Replace attribute names with their values
        for attr, value in sorted(values.items(), key=lambda x: len(x[0]), reverse=True):  # Sort by length to handle longer names first
            right_side = right_side.replace(attr, str(value))
        
        # Replace ^ with ** for exponentiation
        right_side = right_side.replace('^', '**')
        
        # Handle squared terms
        right_side = re.sub(r'(\d+(\.\d+)?)\s*\*\s*(\w+)\s*\*\s*(\w+)', r'\1 * (\3 * \4)', right_side)
        
        # Evaluate the expression
        try:
            result = eval(right_side)
            
            if np.isclose(result, 0, atol=1e-10):
                print("Warning: Result is very close to zero. Check if this is expected.")
            
            return result
        except Exception as e:
            print(f"Error in equation evaluation: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error in calculate_productivity: {str(e)}")
        return None

def calculate_model_sensitivity(response_equation, zscored_statistics):
    try:
        # Get median values for all attributes including the new effective columns
        median_values = {attr: stats['median'] for attr, stats in zscored_statistics.items()}
        
        # Ensure effective_tee and effective_mediandp are in median_values
        if 'effective_tee' not in median_values:
            print("Warning: effective_tee not found in statistics")
            # Calculate effective_tee median from other medians
            if 'tee' in median_values and 'total_slurry_dp' in median_values:
                median_values['effective_tee'] = median_values['tee'] / median_values['total_slurry_dp']
            else:
                median_values['effective_tee'] = 0
                
        if 'effective_mediandp' not in median_values:
            print("Warning: effective_mediandp not found in statistics")
            # Calculate effective_mediandp median from other medians
            if 'median_dp' in median_values and 'median_slurry' in median_values:
                median_values['effective_mediandp'] = median_values['median_dp'] / median_values['median_slurry']
            else:
                median_values['effective_mediandp'] = 0
        
        # Calculate baseline productivity using median values
        baseline_productivity = calculate_productivity(median_values, response_equation)
        
        if baseline_productivity is None:
            print(f"Error: Unable to calculate baseline productivity with equation: {response_equation}")
            return None, None
            
        # Initialize sensitivity results
        sensitivity_results = []
        
        # Define the percentage change for sensitivity analysis
        percent_change = 0.1  # 10% change
        
        # List of attributes to analyze
        attributes = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 
                     'total_slurry_dp', 'median_slurry', 'effective_tee', 'effective_mediandp']
        
        for attr in attributes:
            if attr in zscored_statistics:
                # Get the standard deviation for the attribute
                std_dev = zscored_statistics[attr]['std']
                
                # Calculate the change amount
                change_amount = std_dev * percent_change
                
                # Test increased value
                test_values_high = median_values.copy()
                test_values_high[attr] = median_values[attr] + change_amount
                productivity_high = calculate_productivity(test_values_high, response_equation)
                
                # Test decreased value
                test_values_low = median_values.copy()
                test_values_low[attr] = median_values[attr] - change_amount
                productivity_low = calculate_productivity(test_values_low, response_equation)
                
                if productivity_high is not None and productivity_low is not None:
                    # Calculate sensitivity metrics
                    productivity_change = abs(productivity_high - productivity_low)
                    percent_impact = (productivity_change / baseline_productivity) * 100 if baseline_productivity != 0 else 0
                    
                    sensitivity_results.append({
                        'Attribute': attr,
                        'Min Value': median_values[attr] - change_amount,
                        'Max Value': median_values[attr] + change_amount,
                        'Min Productivity': min(productivity_low, productivity_high),
                        'Max Productivity': max(productivity_low, productivity_high),
                        'Impact': percent_impact
                    })
        
        if not sensitivity_results:
            print("No valid sensitivity results calculated")
            return None, None
            
        # Convert results to DataFrame and sort by absolute impact
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df = sensitivity_df.sort_values('Impact', ascending=False)
        
        return baseline_productivity, sensitivity_df
        
    except Exception as e:
        print(f"Error: Unable to calculate baseline productivity. Equation: {response_equation}")
        print(f"Median values: {median_values}")
        print(f"Error in equation evaluation: {str(e)}")
        return None, None

# function for sensitivity test 
def perform_sensitivity_test(response_equation, attribute, num_points, zscored_statistics):
    # Get the min and max values for the attribute from zscored statistics
    min_value = zscored_statistics[attribute]['min']
    max_value = zscored_statistics[attribute]['max']
    
    # Generate equally spaced test points
    test_points = np.linspace(min_value, max_value, num_points)
    
    # Get median values for all attributes from zscored statistics
    median_values = {attr: zscored_statistics[attr]['median'] for attr in zscored_statistics}
    
    # Calculate productivity for each test point
    results = []
    for point in test_points:
        test_values = median_values.copy()
        test_values[attribute] = point
        productivity = calculate_productivity(test_values, response_equation)
        
        results.append({
            'TestPoint': point,
            'Productivity': productivity
        })
    
    return pd.DataFrame(results)

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
    valid_features = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 'total_slurry_dp', 'median_slurry', 'effective_tee', 'effective_mediandp']
    
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