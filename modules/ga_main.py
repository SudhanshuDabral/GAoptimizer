import streamlit as st
import pandas as pd
import sys
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from utils.db import get_well_details, get_modeling_data, get_well_stages, get_array_data
import re

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ga.ga_calculation as ga_calculation
from ga.check_monotonicity import check_monotonicity as check_monotonicity_func
from utils.plotting import plot_column
import time

def initialize_state():
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'edited_df' not in st.session_state:
        st.session_state.edited_df = None
    if 'zscored_df' not in st.session_state:
        st.session_state.zscored_df = None
    if 'excluded_rows' not in st.session_state:
        st.session_state.excluded_rows = []
    if 'show_zscore_tab' not in st.session_state:
        st.session_state.show_zscore_tab = False
    if 'regression_type' not in st.session_state:
        st.session_state.regression_type = 'FPR'
    if 'monotonicity_results' not in st.session_state:
            st.session_state.monotonicity_results = None

def main():
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.warning("Please login to access this page.")
        st.stop()

    initialize_state()
    st.title("Genetic Algorithm Optimizer for Regression Models (GA-ORM)")

    # Fetch well details from the database
    wells = get_well_details()
    well_options = {well['well_name']: well['well_id'] for well in wells}
    selected_well = st.selectbox("Select a Well", options=list(well_options.keys()))
    well_id = well_options[selected_well]

    # Fetch modeling data for the selected well
    data = get_modeling_data(well_id)
    df = pd.DataFrame(data)
    df = df.sort_values(by='stage')
    df['Productivity'] = ""

    # Create initial tabs
    if st.session_state.show_zscore_tab:
        tab1, tab2 = st.tabs(["Data Preview", "Z-Score Data"])
    else:
        tab1, = st.tabs(["Data Preview"])

    with tab1:
        st.write("Data Preview (You can edit the Productivity column):")

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column("Productivity", editable=True)
        for col in df.columns:
            if col != "Productivity":
                gb.configure_column(col, editable=False)
        gb.configure_grid_options(domLayout='normal', suppressMovableColumns=True, enableRangeSelection=True, clipboardDelimiter=',')

        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            height=400,
            allow_unsafe_jscode=True,
        )

        edited_df = pd.DataFrame(grid_response['data'])
        st.session_state.edited_df = edited_df
        # Calculate and store statistics
        st.session_state.df_statistics = calculate_df_statistics(edited_df)

        if st.button("Z-Score Data"):
            if edited_df['Productivity'].isnull().any() or (edited_df['Productivity'] == "").any():
                st.error("Please ensure all values in the 'Productivity' column are filled.")
            else:
                zscored_df = zscore_data(edited_df)
                st.session_state.zscored_df = zscored_df
                st.session_state.show_zscore_tab = True
                st.success("Data has been Z-Scored.")
                st.rerun()

    if st.session_state.show_zscore_tab:
        with tab2:
            st.write("Z-Scored Data Preview:")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.zscored_df)
            gb.configure_selection('multiple', use_checkbox=True)
            grid_options = gb.build()

            grid_response = AgGrid(
                st.session_state.zscored_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                height=400,
                allow_unsafe_jscode=True,
            )

            selected_rows = pd.DataFrame(grid_response['selected_rows'])
            st.session_state.excluded_rows = selected_rows.index.tolist()

            st.write("Selected rows to exclude from GA optimization:")
            st.dataframe(selected_rows, use_container_width=True, hide_index=True)

    # Custom HTML to style expander headers
    st.markdown(
        """
        <style>
        .streamlit-expanderHeader {
            font-size: 4.5em;
            font-weight: bold;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Feature Selection Section
    with st.expander("Feature Selection"):
        drop_columns = st.multiselect(
            "Select Columns to Drop",
            [col for col in df.columns if col not in ['Productivity', 'stage']],
            help="Choose columns that you do not want to include in the optimization process."
        )
        predictors = [col for col in df.columns if col not in ['Productivity', 'stage'] and col not in drop_columns]

    # GA Optimizer Parameters Section
    with st.expander("GA Optimizer Parameters", expanded=True):
        r2_threshold = st.number_input(
            "R² Threshold",
            min_value=0.0, max_value=1.0, value=0.55,
            help="Set the minimum R² value for model acceptance."
        )
        coef_range = st.slider(
            "Coefficient Range",
            -20.0, 20.0, (-10.0, 10.0),
            help="Select the range for the model coefficients."
        )
        prob_crossover = st.number_input(
            "Crossover Probability",
            min_value=0.0, max_value=1.0, value=0.8,
            help="Set the probability of crossover during genetic algorithm."
        )
        prob_mutation = st.number_input(
            "Mutation Probability",
            min_value=0.0, max_value=1.0, value=0.2,
            help="Set the probability of mutation during genetic algorithm."
        )
        num_generations = st.number_input(
            "Number of Generations",
            min_value=1, value=40,
            help="Specify the number of generations for the genetic algorithm to run."
        )
        population_size = st.number_input(
            "Population Size",
            min_value=1, value=50,
            help="Set the size of the population for the genetic algorithm."
        )
        
        # New input for regression type
        regression_type = st.selectbox(
            "Regression Type",
            options=["Full Polynomial Regression", "Linear with Interaction Parameters"],
            index=0,
            help="Select the type of regression model to use in the optimization process."
        )
        st.session_state.regression_type = 'FPR' if regression_type == "Full Polynomial Regression" else 'LWIP'

    # Main Buttons for GA Optimizer
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start GA Optimization", key="start_button", disabled=st.session_state.running):
            with st.spinner('Running Genetic Algorithm...'):
                st.session_state.running = True
                st.rerun()

    with col2:
        if st.button("Stop GA Optimization", key="stop_button", disabled=not st.session_state.running):
            st.session_state.running = False
            st.rerun()

    if st.session_state.running:
        start_ga_optimization(st.session_state.zscored_df, 'Productivity', predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, st.session_state.excluded_rows, st.session_state.regression_type)

    if st.session_state.result:
        best_ind, best_r2_score, response_equation, selected_feature_names, errors_df = st.session_state.result
        st.success("Genetic Algorithm Optimization Complete!")

        # Display R² Score
        st.subheader("R² Score Achieved")
        st.code(f"{best_r2_score:.4f}", language='text')

        # Display Response Equation
        st.subheader("Response Equation")
        st.code(response_equation, language='text')

        # Display Selected Features
        st.subheader("Selected Features")
        features_text = "\n".join([f"• {feature}" for feature in selected_feature_names])
        st.code(features_text, language="markdown")

        # Display Error Table for Individual Data Points
        st.subheader("Error Table for Individual Data Points")
        st.dataframe(errors_df, use_container_width=True, hide_index=True)

        # Add download button for results
        with pd.ExcelWriter('genetic_algorithm_results.xlsx') as writer:
            pd.DataFrame([best_ind]).to_excel(writer, sheet_name='Best Individual')
            pd.DataFrame([{'R² Score': best_r2_score}]).to_excel(writer, sheet_name='R2 Score')
            pd.DataFrame([{'Response Equation': response_equation}]).to_excel(writer, sheet_name='Response Equation')
            pd.DataFrame(selected_feature_names, columns=['Selected Features']).to_excel(writer, sheet_name='Selected Features')
            errors_df.to_excel(writer, sheet_name='Errors')

        with open('genetic_algorithm_results.xlsx', 'rb') as file:
            st.download_button(
                label="Download GA Results",
                data=file,
                file_name='genetic_algorithm_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        # New section for checking monotonicity
        st.subheader("Check Monotonicity")

        # Get well details and create dropdown
        wells = get_well_details()
        well_options = {well['well_name']: well['well_id'] for well in wells}
        selected_well = st.selectbox("Select a Well", options=list(well_options.keys()), key="monotonicity_well_select")
        well_id = well_options[selected_well]

        # Get stages for selected well and create dropdown
        stages = get_well_stages(well_id)
        selected_stage = st.selectbox("Select a Stage", options=stages, key="monotonicity_stage_select")

        # Custom equation input
        use_custom_equation = st.checkbox("Use custom equation")

        if use_custom_equation:
            custom_equation = st.text_area(
                "Enter custom equation", 
                placeholder="Corrected_Prod = ... (use only tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry)",
                height=100
            )
            
            if custom_equation:
                equation_to_use = custom_equation
                st.info("Using the custom equation for monotonicity check.")
            else:
                st.warning("Please enter a custom equation or uncheck the 'Use custom equation' box.")
                equation_to_use = None
        else:
            equation_to_use = response_equation
            st.info(f"Using the equation from GA optimization: {equation_to_use}")


        # Add button for checking monotonicity
        if st.button("Check Monotonicity"):
            if equation_to_use is None:
                st.error("Please provide a valid equation before checking monotonicity.")
            else:
                # Fetch array data
                array_data = get_array_data(well_id, selected_stage)
                if array_data:
                    # Convert array_data to DataFrame if it's not already
                    if not isinstance(array_data, pd.DataFrame):
                        array_data = pd.DataFrame(array_data)

                    # Perform monotonicity check
                    result_df = check_monotonicity_func(array_data, st.session_state.df_statistics, equation_to_use)

                    # Store results in session state
                    st.session_state.monotonicity_results = result_df
                    st.success("Monotonicity check completed successfully!")
                else:
                    st.error("No array data found for the selected well and stage.")

        # Display results if available
        if st.session_state.monotonicity_results is not None:
            st.subheader("Monotonicity Check Results")
            st.write(f"Total rows: {len(st.session_state.monotonicity_results)}")
            st.dataframe(st.session_state.monotonicity_results, use_container_width=True, height=400)

            # Option to download the results
            csv = st.session_state.monotonicity_results.to_csv(index=False)
            st.download_button(
                label="Download Monotonicity Results",
                data=csv,
                file_name=f"monotonicity_results_well_{selected_well}_stage_{selected_stage}.csv",
                mime="text/csv",
            )

            st.subheader("Plot Monotonicity Results")

            # Multi-select dropdown for column selection
            plot_columns = st.multiselect(
                "Select columns to plot",
                options=st.session_state.monotonicity_results.columns.tolist(),
                default=["Productivity"]
            )

            # Plot each selected column
            for column in plot_columns:
                st.plotly_chart(plot_column(st.session_state.monotonicity_results, column))
        else:
            st.info("Run the monotonicity check to see results and plots.")

def start_ga_optimization(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, excluded_rows, regression_type):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.drop(excluded_rows)
    start_time = time.time()
    timer_placeholder = st.empty()

    result = ga_calculation.run_ga(
        df, target_column, predictors, r2_threshold, coef_range,
        prob_crossover, prob_mutation, num_generations, population_size,
        timer_placeholder, regression_type
    )

    if result:
        best_ind, best_r2_score, response_equation, selected_feature_names, errors_df = result
        st.session_state.result = (best_ind, best_r2_score, response_equation, selected_feature_names, errors_df)
    else:
        st.session_state.result = None

    st.session_state.running = False
    st.rerun()

def zscore_data(df):
    for column in df.columns:
        if column != 'stage':
            df[column] = pd.to_numeric(df[column], errors='coerce')
            col_mean = df[column].mean()
            col_std = df[column].std(ddof=1)
            df[column] = (df[column] - col_mean) / col_std
    return df

# function to calculate statistics for the original data for modelling 
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

# function to validate custom equation
# def validate_custom_equation(equation):
#     valid_features = ['tee', 'median_dhpm', 'median_dp', 'downhole_ppm', 'total_dhppm', 'total_slurry_dp', 'median_slurry']
    
#     # Check if equation starts with "Corrected_Prod ="
#     if not equation.strip().startswith("Corrected_Prod ="):
#         return False, "Equation must start with 'Corrected_Prod ='"
    
#     # Remove "Corrected_Prod =" from the equation for further processing
#     equation = equation.replace("Corrected_Prod =", "").strip()
    
#     # Replace feature names with placeholder variables
#     for feature in valid_features:
#         equation = re.sub(r'\b' + feature + r'\b', 'x', equation)
    
#     # Handle squared terms
#     equation = re.sub(r'(\w+)\^2', r'\1 * \1', equation)
    
#     # Remove all spaces
#     equation = equation.replace(" ", "")
    
#     # Check for valid characters
#     valid_chars = set('x+-*/().')
#     if not all(char in valid_chars for char in equation):
#         return False, "Equation contains invalid characters"
    
#     # Check for balanced parentheses
#     if equation.count('(') != equation.count(')'):
#         return False, "Unbalanced parentheses in equation"
    
#     # Try to evaluate the equation
#     try:
#         x = 1  # Dummy value for testing
#         eval(equation)
#     except:
#         return False, "Invalid equation structure"
    
#     return True, "Equation is valid"

if __name__ == "__main__":
    main()