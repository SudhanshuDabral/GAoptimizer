import warnings
import contextlib
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from utils.db import (get_well_details, get_modeling_data,
                       get_well_stages, get_array_data, insert_ga_model)
from logging.handlers import RotatingFileHandler
import logging
from ga.ga_calculation import run_ga as run_ga_optimization
from ga.check_monotonicity import check_monotonicity as check_monotonicity_func
from utils.plotting import (plot_column, plot_actual_vs_predicted, create_tornado_chart,
                            create_feature_importance_chart, create_elasticity_analysis, 
                            plot_sensitivity_results, create_influence_chart)
from utils.ga_utils import (zscore_data, calculate_df_statistics,
                             validate_custom_equation, calculate_predicted_productivity, 
                             calculate_model_sensitivity, calculate_zscoredf_statistics, perform_sensitivity_test)

from utils.reporting import generate_pdf_report, generate_ppt_report
import time
import os
import numpy as np


# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'app_logs.log')

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
                        ])

logger = logging.getLogger(__name__)

def log_message(level, message):
    logger.log(level, f"[GA Optimizer] {message}")

def initialize_ga_state():
    if 'ga_optimizer' not in st.session_state:
        st.session_state.ga_optimizer = {
            'running': False,
            'results': [],
            'edited_df': None,
            'zscored_df': None,
            'excluded_rows': [],
            'show_zscore': False,
            'regression_type': 'FPR',
            'monotonicity_results': {},
            'df_statistics': None,
            'zscored_statistics': None,
            'show_monotonicity': False,
            'show_sensitivity_test': False,
            'r2_threshold': 0.55,
            'prob_crossover': 0.8,
            'prob_mutation': 0.2,
            'num_generations': 40,
            'population_size': 50,
            'predictors': [],
            'selected_wells': [],  # Add this line
            'coef_range': (-10.0, 10.0),  # Add this line
            'num_models': 3,  # Add this line
        }

def start_ga_optimization_callback():
    st.session_state.ga_optimizer['running'] = True
    st.session_state.ga_optimizer['results'] = [] 
    st.session_state.continuous_optimization = {
        'r2_values': [],
        'iterations': [],
        'model_markers': {},
        'current_iteration': 0
    }


# GA optimization main function
def main(authentication_status):
    if not authentication_status:
        st.warning("Please login to access this page.")
        log_message(logging.WARNING, "User not authenticated")
        return

    initialize_ga_state()

    st.sidebar.title("GA Optimizer Menu")
    if st.sidebar.button("Monotonicity Check", key="fab_monotonicity"):
        st.session_state.ga_optimizer['show_monotonicity'] = not st.session_state.ga_optimizer.get('show_monotonicity', False)
    
    if st.sidebar.button("Sensitivity Test", key="fab_sensitivity_test"):
        st.session_state.ga_optimizer['show_sensitivity_test'] = not st.session_state.ga_optimizer.get('show_sensitivity_test', False)
    
    st.title("Hydraulic Fracturing Productivity Model Optimizer (HF-PMO)")

    ga_optimization_section()

    if st.session_state.ga_optimizer.get('show_monotonicity', False):
        monotonicity_check_modal()
    
    if st.session_state.ga_optimizer.get('show_sensitivity_test', False):
        sensitivity_test_section()


@contextlib.contextmanager
def suppress_st_aggrid_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed file")
        yield

def ga_optimization_section():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        try:
            wells = get_well_details()
            well_options = {well['well_name']: well['well_id'] for well in wells}
            
            selected_wells = st.multiselect("Select Wells", options=list(well_options.keys()), key="ga_well_select")
            st.session_state.ga_optimizer['selected_wells'] = selected_wells  # Store selected wells
            selected_well_ids = [well_options[well] for well in selected_wells]
            
            if not selected_wells:
                st.warning("Please select at least one well to proceed.")
                return

            # Add a button to load data for all selected wells
            if st.button("Load Data for Selected Wells"):
                with st.spinner("Loading data for selected wells..."):
                    consolidated_data = fetch_consolidated_data(selected_well_ids)
                    
                    if consolidated_data.empty:
                        st.error("No data available for the selected wells.")
                        return

                    df = consolidated_data.sort_values(by=['Well Name', 'stage'])
                    
                    # Initialize Productivity column if it doesn't exist
                    if 'Productivity' not in df.columns:
                        df['Productivity'] = ""
                    
                    st.session_state.ga_optimizer['edited_df'] = df
                    st.success(f"Data loaded for {len(selected_wells)} well(s).")

            # Display the AgGrid only if data has been loaded
            if 'edited_df' in st.session_state.ga_optimizer and st.session_state.ga_optimizer['edited_df'] is not None:
                df = st.session_state.ga_optimizer['edited_df']

                if st.session_state.ga_optimizer['show_zscore']:
                    tab1, tab2 = st.tabs(["Data Preview", "Z-Score Data"])
                else:
                    tab1, = st.tabs(["Data Preview"])

                with tab1:
                    st.write("Data Preview (You can edit the Productivity column):")
                    gb = GridOptionsBuilder.from_dataframe(df)
                    gb.configure_column("Productivity", editable=True)
                    gb.configure_column("Well Name", hide=False)
                    gb.configure_column("data_id", hide=True)
                    gb.configure_column("well_id", hide=True)
                    for col in df.columns:
                        if col not in ['Productivity', 'Well Name', 'data_id', 'well_id']:
                            gb.configure_column(col, editable=False)
                    gb.configure_grid_options(domLayout='normal', 
                                              suppressMovableColumns=True, 
                                              enableRangeSelection=True, 
                                              clipboardDelimiter=',',
                                              columnSizeDefault=150,
                                              autoSizeColumns=True)
                    grid_options = gb.build()
                    
                    grid_response = AgGrid(df, 
                                           gridOptions=grid_options, 
                                           update_mode=GridUpdateMode.VALUE_CHANGED,
                                           fit_columns_on_grid_load=True,
                                           height=400, 
                                           allow_unsafe_jscode=True)
                    
                    edited_df = pd.DataFrame(grid_response['data'])
                    st.session_state.ga_optimizer['edited_df'] = edited_df
                    st.session_state.ga_optimizer['df_statistics'] = calculate_df_statistics(edited_df)

                    # Add download button for data preview
                    st.download_button(
                        label="Download Data Preview",
                        data=edited_df.to_csv(index=False).encode('utf-8'),
                        file_name="data_preview.csv",
                        mime="text/csv",
                    )

                    if st.button("Z-Score Data"):
                        if edited_df['Productivity'].isnull().any() or (edited_df['Productivity'] == "").any():
                            st.error("Please ensure all values in the 'Productivity' column are filled.")
                        else:
                            zscored_df = zscore_data(edited_df)
                            st.session_state.ga_optimizer['zscored_df'] = zscored_df
                            st.session_state.ga_optimizer['zscored_statistics'] = calculate_zscoredf_statistics(st.session_state.ga_optimizer['zscored_df'])
                            st.session_state.ga_optimizer['show_zscore'] = True
                            st.success("Data has been Z-Scored.")
                            st.rerun()

                if st.session_state.ga_optimizer['show_zscore']:
                    with tab2:
                        st.write("Z-Scored Data Preview:")
                        gb = GridOptionsBuilder.from_dataframe(st.session_state.ga_optimizer['zscored_df'])
                        gb.configure_selection('multiple', use_checkbox=True)
                        gb.configure_column("Well Name", hide=False)
                        gb.configure_column("data_id", hide=True)
                        gb.configure_column("well_id", hide=True)
                        gb.configure_column("tee", checkboxSelection=True, headerCheckboxSelection=True)
                        gb.configure_grid_options(suppressRowClickSelection=True,
                                                  columnSizeDefault=150,  # Set default column width
                                                  autoSizeColumns=True)  # Enable auto-sizing
                        grid_options = gb.build()
                        
                        grid_response = AgGrid(st.session_state.ga_optimizer['zscored_df'], 
                                               gridOptions=grid_options,
                                               update_mode=GridUpdateMode.SELECTION_CHANGED, 
                                               fit_columns_on_grid_load=True,  # Fit columns on load
                                               height=400, 
                                               allow_unsafe_jscode=True)
                        
                        selected_rows = pd.DataFrame(grid_response['selected_rows'])
                        
                        if not selected_rows.empty:
                            if 'data_id' in selected_rows.columns:
                                st.session_state.ga_optimizer['excluded_rows'] = selected_rows['data_id'].tolist()
                            else:
                                # If data_id is not in selected_rows, we need to map the selections back to the original dataframe
                                selected_indices = [st.session_state.ga_optimizer['zscored_df'].index.get_loc(row['Well Name']) for row in selected_rows.to_dict('records')]
                                st.session_state.ga_optimizer['excluded_rows'] = st.session_state.ga_optimizer['zscored_df'].iloc[selected_indices]['data_id'].tolist()
                        else:
                            st.session_state.ga_optimizer['excluded_rows'] = []

                        st.download_button(
                            label="Download Z-Scored Data",
                            data=st.session_state.ga_optimizer['zscored_df'].to_csv(index=False).encode('utf-8'),
                            file_name="zscored_data.csv",
                            mime="text/csv",
                        )
                        
                        st.write("Selected rows to exclude from GA optimization:")
                        display_columns = [col for col in selected_rows.columns if col not in ['data_id', 'well_id']]
                        st.dataframe(selected_rows[display_columns], use_container_width=True, hide_index=True)
                        
                        # Display the number of selected rows
                        st.write(f"Number of datapoints selected for exclusion: {len(st.session_state.ga_optimizer['excluded_rows'])}")

                with st.expander("Feature Selection"):
                    drop_columns = st.multiselect("Select Columns to Drop",
                                                  [col for col in df.columns if col not in ['Productivity', 'stage', 'data_id', 'well_id', 'Well Name']],
                                                  help="Choose columns that you do not want to include in the optimization process.")
                    predictors = [col for col in df.columns if col not in ['Productivity', 'stage', 'Well Name', 'data_id', 'well_id'] and col not in drop_columns]

                with st.expander("GA Optimizer Parameters", expanded=True):
                    st.session_state.ga_optimizer['r2_threshold'] = st.number_input("R² Threshold", min_value=0.0, max_value=1.0, value=0.55,
                                                   help="Set the minimum R² value for model acceptance.")
                    coef_range = st.slider("Coefficient Range", -20.0, 20.0, (-10.0, 10.0),
                                           help="Select the range for the model coefficients.")
                    st.session_state.ga_optimizer['prob_crossover'] = st.number_input("Crossover Probability", min_value=0.0, max_value=1.0, value=0.8,
                                                     help="Set the probability of crossover during genetic algorithm.")
                    st.session_state.ga_optimizer['prob_mutation'] = st.number_input("Mutation Probability", min_value=0.0, max_value=1.0, value=0.2,
                                                   help="Set the probability of mutation during genetic algorithm.")
                    st.session_state.ga_optimizer['num_generations'] = st.number_input("Number of Generations", min_value=1, value=40,
                                                      help="Specify the number of generations for the genetic algorithm to run.")
                    st.session_state.ga_optimizer['population_size'] = st.number_input("Population Size", min_value=1, value=50,
                                                      help="Set the size of the population for the genetic algorithm.")
                    num_models = st.number_input("Number of Models to Generate", min_value=1, max_value=6, value=3,
                                                 help="Specify the number of models to generate that meet the R² threshold.")
                    regression_type = st.selectbox("Regression Type",
                                                   options=["Full Polynomial Regression", "Linear with Interaction Parameters"],
                                                   index=0,
                                                   help="Select the type of regression model to use in the optimization process.")
                    st.session_state.ga_optimizer['regression_type'] = 'FPR' if regression_type == "Full Polynomial Regression" else 'LWIP'
                    

                if st.session_state.ga_optimizer['running']:
                    if st.button("Stop GA Optimization", key="toggle_button"):
                        st.session_state.ga_optimizer['running'] = False
                        st.rerun()
                else:
                    if st.button("Start GA Optimization", key="toggle_button", on_click=start_ga_optimization_callback):
                        with st.spinner('Running Genetic Algorithm...'):
                            start_ga_optimization(st.session_state.ga_optimizer['zscored_df'], 'Productivity', predictors, st.session_state.ga_optimizer['r2_threshold'],
                                                coef_range, st.session_state.ga_optimizer['prob_crossover'], st.session_state.ga_optimizer['prob_mutation'], 
                                                st.session_state.ga_optimizer['num_generations'], st.session_state.ga_optimizer['population_size'],
                                                st.session_state.ga_optimizer['excluded_rows'],
                                                st.session_state.ga_optimizer['regression_type'], num_models)

                if st.session_state.ga_optimizer['running']:
                    start_ga_optimization(st.session_state.ga_optimizer['zscored_df'], 'Productivity', predictors, st.session_state.ga_optimizer['r2_threshold'],
                                          coef_range, st.session_state.ga_optimizer['prob_crossover'], st.session_state.ga_optimizer['prob_mutation'], 
                                          st.session_state.ga_optimizer['num_generations'], st.session_state.ga_optimizer['population_size'],
                                          st.session_state.ga_optimizer['excluded_rows'],
                                          st.session_state.ga_optimizer['regression_type'], num_models)

                if st.session_state.ga_optimizer['results']:
                    display_ga_results()

        except Exception as e:
            log_message(logging.ERROR, f"Error in ga_optimization_section: {str(e)}")
            st.error("An error occurred during the GA optimization process. Please check the logs for more information.")

        # Log any warnings that were caught
        for warning in caught_warnings:
            if not any(ignored_message in str(warning.message) for ignored_message in ["unclosed file", "fit_columns_on_grid_load is deprecated"]):
                log_message(logging.WARNING, f"Warning in ga_optimization_section: {warning.message}")

    log_message(logging.INFO, "Finished GA optimization section")


def fetch_consolidated_data(well_ids):
    consolidated_data = pd.DataFrame()
    try:
        for well_id in well_ids:
            data = get_modeling_data(well_id)
            df = pd.DataFrame(data)
            if not df.empty:
                well_name = next(well['well_name'] for well in get_well_details() if well['well_id'] == well_id)
                df['Well Name'] = well_name
                consolidated_data = pd.concat([consolidated_data, df], ignore_index=True)
            else:
                log_message(logging.WARNING, f"No data available for well_id: {well_id}")
        return consolidated_data
    except Exception as e:
        log_message(logging.ERROR, f"Error in fetch_consolidated_data: {str(e)}")
        raise

def display_ga_results():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        try:
            st.success(f"Genetic Algorithm Optimization Complete! Generated {len(st.session_state.ga_optimizer['results'])} models.")

            for i, result in enumerate(st.session_state.ga_optimizer['results']):
                best_ind, weighted_r2_score, response_equation, selected_feature_names, errors_df, predicted_values, zscored_df, excluded_rows, full_dataset_r2 = result
                 
                st.subheader(f"Model {i+1}")
                st.write(f"Weighted R² Score (Train/Test): {weighted_r2_score:.4f}")
                st.write(f"Full Dataset R² Score: {full_dataset_r2:.4f}")
                st.code(response_equation, language='text')
                
                with st.expander("Show Details"):
                    st.write("Selected Features:")
                    features_text = "\n".join([f"• {feature}" for feature in selected_feature_names])
                    st.code(features_text, language="markdown")
                    
                    # Add dropdown for selecting percentage of top error points
                    error_percentage = st.selectbox(
                        "Select percentage of top error points to exclude",
                        options=[5, 10, 15, 18, 20, 22, 25, 30],
                        format_func=lambda x: f"{x}%",
                        key=f"error_percentage_{i}"
                    )

                    # Button to apply the exclusion
                    if st.button("Apply Exclusion", key=f"apply_exclusion_{i}"):
                        # Sort errors_df by absolute error
                        errors_df['Abs_Error'] = abs(errors_df['Error'])
                        sorted_errors = errors_df.sort_values('Abs_Error', ascending=False)
                        
                        # Calculate number of points to exclude
                        num_points = int(len(errors_df) * error_percentage / 100)
                        
                        # Get data_ids of points to exclude
                        new_excluded_ids = sorted_errors.head(num_points)['data_id'].tolist()
                        
                        # Update excluded_rows
                        st.session_state.ga_optimizer['excluded_rows'] = list(set(excluded_rows + new_excluded_ids))
                        
                        # Update zscored table checkboxes
                        if 'zscored_df' in st.session_state.ga_optimizer:
                            st.session_state.ga_optimizer['zscored_df']['excluded'] = st.session_state.ga_optimizer['zscored_df']['data_id'].isin(st.session_state.ga_optimizer['excluded_rows'])
                        
                        st.success(f"Added {num_points} points with largest errors to excluded rows.")
                        st.rerun()
                    
                    st.write("Error Table for Individual Data Points")
                    errors_df_display = errors_df.drop(columns=['data_id', 'well_id'])
                    st.dataframe(errors_df_display, use_container_width=True, hide_index=True)

                    st.write("Actual vs Predicted Productivity Plot")
                    fig = plot_actual_vs_predicted(errors_df)
                    st.plotly_chart(fig, use_container_width=True)

                    # Model Sensitivity Analysis
                    st.write("Model Sensitivity Analysis")
                    try:
                        baseline_productivity, sensitivity_df = calculate_model_sensitivity(response_equation, st.session_state.ga_optimizer['zscored_statistics'])
                        
                        if baseline_productivity is not None and not np.isclose(baseline_productivity, 0, atol=1e-10):
                            st.write(f"Baseline Productivity (using median values): {baseline_productivity:.4f}")
                            
                            st.write("Sensitivity Analysis Results:")
                            st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)
                            
                            if not sensitivity_df['Min Productivity'].isna().all() and not sensitivity_df['Max Productivity'].isna().all():
                                fig = create_tornado_chart(sensitivity_df, baseline_productivity)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Unable to create tornado chart due to invalid productivity values.")
                        else:
                            st.error("Baseline productivity is zero or None. Please check the model equation and input values.")


                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            show_more_visuals = st.button(f"Show More Visuals", key=f"more_visuals_{i}")
                        
                        with col2:
                            model_name = st.text_input("Model Name", key=f"model_name_{i}", placeholder="Enter model name")
                        
                        with col3:
                            save_model = st.button("Save Model", key=f"save_model_{i}", disabled=not model_name)

                        if show_more_visuals:
                            st.write("Feature Importance Chart")
                            st.write("This chart shows the relative importance of each feature based on its impact on productivity.")
                            fig_importance = create_feature_importance_chart(sensitivity_df)
                            st.plotly_chart(fig_importance, use_container_width=True)

                            st.write("Elasticity Analysis")
                            st.write("This chart shows how sensitive the productivity is to changes in each feature.")
                            fig_elasticity = create_elasticity_analysis(sensitivity_df, st.session_state.ga_optimizer['zscored_statistics'], baseline_productivity)
                            st.plotly_chart(fig_elasticity, use_container_width=True)

                        if save_model:
                            # Prepare data for saving
                            ga_params = {
                                'r2_threshold': st.session_state.ga_optimizer['r2_threshold'],
                                'prob_crossover': st.session_state.ga_optimizer['prob_crossover'],
                                'prob_mutation': st.session_state.ga_optimizer['prob_mutation'],
                                'num_generations': st.session_state.ga_optimizer['num_generations'],
                                'population_size': st.session_state.ga_optimizer['population_size'],
                                'regression_type': st.session_state.ga_optimizer['regression_type'],
                                'feature_selection': selected_feature_names
                            }
                            ga_results = {
                                'response_equation': response_equation,
                                'best_r2_score': weighted_r2_score,
                                'full_dataset_r2': full_dataset_r2,
                                'selected_feature_names': selected_feature_names
                            }
                             # Get the predictors from the session state
                            predictors = st.session_state.ga_optimizer.get('predictors', [])

                            # Call the database function to save the model
                            result = insert_ga_model(
                                model_name,
                                st.session_state.user_id,
                                ga_params,
                                ga_results,
                                zscored_df,
                                excluded_rows,
                                sensitivity_df,
                                st.session_state.ga_optimizer['zscored_statistics'],
                                baseline_productivity,
                                predictors
                                )
                            
                            if result['status'] == 'success':
                                st.success(f"Model '{model_name}' saved successfully.")
                            elif result['status'] == 'duplicate':
                                st.warning(result['message'])
                            else:
                                st.error(result['message'])

                    except Exception as e:
                        st.error(f"An error occurred during sensitivity analysis: {str(e)}")
                        log_message(logging.ERROR, f"Error in sensitivity analysis: {str(e)}")

            if st.session_state.ga_optimizer['results']:
                with pd.ExcelWriter('genetic_algorithm_results.xlsx') as writer:
                    for i, result in enumerate(st.session_state.ga_optimizer['results']):
                        best_ind, weighted_r2_score, response_equation, selected_feature_names, errors_df, predicted_values, zscored_df, excluded_rows, full_dataset_r2 = result
                        
                        pd.DataFrame([best_ind]).to_excel(writer, sheet_name=f'Model_{i+1}_Best_Individual')
                        pd.DataFrame([{'R² Score': weighted_r2_score, 'Response Equation': response_equation}]).to_excel(writer, sheet_name=f'Model_{i+1}_Details')
                        pd.DataFrame(selected_feature_names, columns=['Selected Features']).to_excel(writer, sheet_name=f'Model_{i+1}_Features')
                        errors_df.to_excel(writer, sheet_name=f'Model_{i+1}_Errors')

                with open('genetic_algorithm_results.xlsx', 'rb') as file:
                    st.download_button(
                        label="Download GA Results",
                        data=file,
                        file_name='genetic_algorithm_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
            else:
                st.warning("No results available to download.")

        except Exception as e:
            log_message(logging.ERROR, f"Error in display_ga_results: {str(e)}")
            st.error("An error occurred while displaying the results. Please check the logs for more information.")

        # Log any warnings that were caught
        for warning in caught_warnings:
            log_message(logging.WARNING, f"Warning in display_ga_results: {warning.message}")

    log_message(logging.INFO, "Finished displaying GA results")

# Monotonicity Check
def monotonicity_check_modal():
    st.markdown("""
    <style>
    .section-partition {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 0;
        border-top: 2px solid #e5e5e5;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-partition">', unsafe_allow_html=True)
    st.markdown("## Monotonicity Check")

    wells = get_well_details()
    well_options = {well['well_name']: well['well_id'] for well in wells}
    selected_well = st.selectbox("Select a Well", options=list(well_options.keys()), key="monotonicity_well_select")
    well_id = well_options[selected_well]

    stages = get_well_stages(well_id)
    
    # Add "Select All" option
    all_stages_option = "Select All Stages"
    stage_options = [all_stages_option] + stages
    
    selected_stages = st.multiselect("Select Stage(s)", options=stage_options, key="monotonicity_stage_select")
    
    # Handle "Select All" option
    if all_stages_option in selected_stages:
        selected_stages = stages

    if 'results' in st.session_state.ga_optimizer and st.session_state.ga_optimizer['results']:
        model_options = [f"Model {i+1} (Weighted R²: {result[1]:.4f}, Full Dataset R²: {result[8]:.4f})" 
                     for i, result in enumerate(st.session_state.ga_optimizer['results'])]
        model_options.append("Custom Equation")
        selected_model = st.selectbox("Select Model for Monotonicity Check", options=model_options)
    else:
        selected_model = "Custom Equation"
        st.info("No GA models available. Using custom equation.")

    if selected_model == "Custom Equation":
        custom_equation = st.text_area(
            "Enter custom equation", 
            placeholder="Corrected_Prod = ... (use only tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry)", key='monotonicity_custom_eqn' ,
            height=100
        )
        
        if custom_equation:
            is_valid, validation_message = validate_custom_equation(custom_equation)
            if is_valid:
                equation_to_use = custom_equation
                st.success("Custom equation is valid and will be used for the monotonicity check.")
            else:
                st.error(f"Invalid custom equation: {validation_message}")
                equation_to_use = None
        else:
            st.warning("Please enter a custom equation.")
            equation_to_use = None
    else:
        model_index = int(selected_model.split()[1]) - 1
        equation_to_use = st.session_state.ga_optimizer['results'][model_index][2]
        st.info(f"Using equation from {selected_model}")

    if equation_to_use:
        st.write("Equation being used for monotonicity check:")
        st.code(equation_to_use)
    
    if st.button("Run Monotonicity Check"):
        if equation_to_use is None:
            st.error("Please provide a valid equation before checking monotonicity.")
        elif not selected_stages:
            st.error("Please select at least one stage for monotonicity check.")
        else:
            run_monotonicity_check(well_id, selected_stages, equation_to_use)        
    
    display_monotonicity_results(selected_stages)

def run_monotonicity_check(well_id, selected_stages, equation_to_use):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = {}
    for i, stage in enumerate(selected_stages):
        status_text.text(f"Checking monotonicity for Stage {stage}...")
        array_data = get_array_data(well_id, stage)
        if array_data is not None:
            if not isinstance(array_data, pd.DataFrame):
                array_data = pd.DataFrame(array_data)
            result_df = check_monotonicity_func(array_data, st.session_state.ga_optimizer['df_statistics'], equation_to_use)
            results[stage] = result_df
        else:
            st.warning(f"No data available for Stage {stage}. Skipping...")
        progress_bar.progress((i + 1) / len(selected_stages))

    st.session_state.ga_optimizer['monotonicity_results'] = results
    status_text.text("Monotonicity check completed!")
    st.success(f"Monotonicity check completed successfully for {len(results)} out of {len(selected_stages)} stages.")

def display_monotonicity_results(selected_stages):
    if st.session_state.ga_optimizer['monotonicity_results']:
        if len(selected_stages) == 1:
            stage = selected_stages[0]
            result_df = st.session_state.ga_optimizer['monotonicity_results'][stage]
            
            st.subheader(f"Monotonicity Check Results for Stage {stage}")
            st.write(f"Total rows: {len(result_df)}")
            st.dataframe(result_df, use_container_width=True, height=400)

            st.subheader("Plot Monotonicity Results")
            plot_columns = st.multiselect(
                "Select columns to plot",
                options=result_df.columns.tolist(),
                default=["Productivity"]
            )
            for column in plot_columns:
                fig = plot_column(result_df, column, stage)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Productivity, Total DHPPM, and Total Slurry DP Plots for Selected Stages")
            for stage, result_df in st.session_state.ga_optimizer['monotonicity_results'].items():
                # Check if the required columns are present
                required_columns = ['Productivity', 'total_dhppm_stage', 'total_slurry_dp_stage']
                missing_columns = [col for col in required_columns if col not in result_df.columns]
                
                if missing_columns:
                    st.warning(f"Stage {stage}: Missing columns: {', '.join(missing_columns)}")
                    st.write("Available columns:", result_df.columns.tolist())
                else:
                    fig = plot_column(result_df, stage)
                    st.plotly_chart(fig, use_container_width=True)

                # Display the first few rows of the DataFrame for this stage
                # st.write(f"Preview of data for Stage {stage}:")
                # st.dataframe(result_df.head())
    else:
        st.info("Run Monotonicity check to see results.")

# Sensitivity Test
def sensitivity_test_section():
    st.markdown("## Sensitivity Test")

    # Initialize session state for storing results
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = {
            'general': None,
            'attribute_specific': None,
            'current_model': None
        }

    # Check for the presence of models and zscored statistics
    if not st.session_state.ga_optimizer['results']:
        st.warning("No models have been generated. You can use a custom equation for sensitivity analysis.")
        model_options = ["Custom Equation"]
    else:
        model_options = [f"Model {i+1} (Weighted R²: {result[1]:.4f}, Full Dataset R²: {result[8]:.4f})" 
                         for i, result in enumerate(st.session_state.ga_optimizer['results'])]
        model_options.append("Custom Equation")

    if 'zscored_statistics' not in st.session_state.ga_optimizer:
        st.error("Z-scored statistics are not available. Please process your data before proceeding with sensitivity analysis.")
        return

    selected_model = st.selectbox("Select Model or Custom Equation", options=model_options, key="sensitivity_model_select")

    # Reset results if model changes
    if selected_model != st.session_state.sensitivity_results.get('current_model'):
        st.session_state.sensitivity_results = {
            'general': None,
            'attribute_specific': None,
            'current_model': selected_model
        }

    if selected_model == "Custom Equation":
        st.info("""
        You've selected to use a custom equation for sensitivity analysis. 
        Please note that while your custom equation will be used for calculations, 
        the sensitivity and applicability of this equation will be tested using 
        the dataset currently loaded in the session. 
        
        Ensure that your equation is compatible with the variables in the current dataset 
        for accurate results.
        """)
        custom_equation = st.text_area(
            "Enter custom equation", 
            placeholder="Corrected_Prod = ... (use only tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry)", 
            key='sensitivity_custom_eqn',
            height=100
        )
        is_valid, validation_message = validate_custom_equation(custom_equation)
        if not is_valid:
            st.error(f"Invalid custom equation: {validation_message}")
            return
        equation_to_use = custom_equation
    else:
        model_index = int(selected_model.split()[1]) - 1
        _, _, equation_to_use, _, _, _, _, _, _ = st.session_state.ga_optimizer['results'][model_index]

    st.write("Equation being used for sensitivity analysis:")
    st.code(equation_to_use, language='text')

    # Create tabs
    tab1, tab2 = st.tabs(["General Sensitivity Analysis", "Attribute-Specific Sensitivity Test"])

    with tab1:
        st.subheader("General Sensitivity Analysis")
        if st.button("Run General Sensitivity Analysis", key="run_general_sensitivity"):
            with st.spinner("Running General Sensitivity Analysis..."):
                try:
                    baseline_productivity, sensitivity_df = calculate_model_sensitivity(
                        equation_to_use, 
                        st.session_state.ga_optimizer['zscored_statistics']
                    )
                    
                    if baseline_productivity is not None and not np.isclose(baseline_productivity, 0, atol=1e-10):
                        fig_tornado = create_tornado_chart(sensitivity_df, baseline_productivity)
                        fig_importance = create_feature_importance_chart(sensitivity_df)
                        fig_elasticity = create_elasticity_analysis(sensitivity_df, st.session_state.ga_optimizer['zscored_statistics'], baseline_productivity)
                        
                        st.session_state.sensitivity_results['general'] = {
                            'baseline_productivity': baseline_productivity,
                            'sensitivity_df': sensitivity_df,
                            'fig_tornado': fig_tornado,
                            'fig_importance': fig_importance,
                            'fig_elasticity': fig_elasticity
                        }
                        st.success("General Sensitivity Analysis completed successfully!")
                    else:
                        st.error("Baseline productivity is zero or None. Please check the equation and input values.")
                except Exception as e:
                    st.error(f"An error occurred during general sensitivity analysis: {str(e)}")
                    log_message(logging.ERROR, f"Error in general sensitivity analysis: {str(e)}")
        
        if st.session_state.sensitivity_results['general']:
            results = st.session_state.sensitivity_results['general']
            st.write(f"Baseline Productivity: {results['baseline_productivity']:.4f}")
            st.dataframe(results['sensitivity_df'], use_container_width=True, hide_index=True)
            st.plotly_chart(results['fig_tornado'], use_container_width=True)
            st.plotly_chart(results['fig_importance'], use_container_width=True)
            st.plotly_chart(results['fig_elasticity'], use_container_width=True)

    with tab2:
        st.subheader("Attribute-Specific Sensitivity Test")
        if st.session_state.sensitivity_results['general']:
            predictors = st.session_state.sensitivity_results['general']['sensitivity_df']['Attribute'].tolist()
            
            # Add "Select All" option
            all_attributes_option = "Select All Attributes"
            attribute_options = [all_attributes_option] + predictors
            
            selected_attributes = st.multiselect("Select Attributes for Detailed Sensitivity Test", 
                                                options=attribute_options, 
                                                key="sensitivity_attributes_select")
            
            # Handle "Select All" option
            if all_attributes_option in selected_attributes:
                selected_attributes = predictors
            
            num_points = st.slider("Number of Test Points", min_value=5, max_value=50, value=20, key="sensitivity_num_points")

            if st.button("Run Attribute-Specific Sensitivity Test", key="run_attribute_sensitivity"):
                with st.spinner("Running Attribute-Specific Sensitivity Test..."):
                    try:
                        attribute_results = {}
                        for attr in selected_attributes:
                            sensitivity_results = perform_sensitivity_test(
                                equation_to_use,
                                attr, 
                                num_points, 
                                st.session_state.ga_optimizer['zscored_statistics']
                            )
                            fig_sensitivity = plot_sensitivity_results(sensitivity_results, attr)
                            
                            prod_min = sensitivity_results['Productivity'].min()
                            prod_max = sensitivity_results['Productivity'].max()
                            prod_range = prod_max - prod_min
                            attr_min = sensitivity_results['TestPoint'].min()
                            attr_max = sensitivity_results['TestPoint'].max()
                            
                            attribute_results[attr] = {
                                'results': sensitivity_results,
                                'fig': fig_sensitivity,
                                'prod_range': prod_range,
                                'prod_min': prod_min,
                                'prod_max': prod_max,
                                'attr_min': attr_min,
                                'attr_max': attr_max
                            }
                        
                        st.session_state.sensitivity_results['attribute_specific'] = attribute_results
                        st.success("Attribute-Specific Sensitivity Test completed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during attribute-specific sensitivity test: {str(e)}")
                        log_message(logging.ERROR, f"Error in attribute-specific sensitivity test: {str(e)}")
            
            if st.session_state.sensitivity_results['attribute_specific']:
                results = st.session_state.sensitivity_results['attribute_specific']
                
                # Consolidated view of attribute influence
                st.subheader("Consolidated Attribute Influence")
                influence_data = [
                    {"Attribute": attr, "Productivity Range": data['prod_range'], "Influence": data['prod_range']}
                    for attr, data in results.items() if attr not in ['influence_df', 'fig_influence']
                ]
                influence_df = pd.DataFrame(influence_data)
                influence_df = influence_df.sort_values('Influence', ascending=False)
                
                # Display influence chart
                fig_influence = create_influence_chart(influence_df)
                st.plotly_chart(fig_influence, use_container_width=True)
                
                # Store the influence data and chart in the session state
                st.session_state.sensitivity_results['attribute_specific']['influence_df'] = influence_df
                st.session_state.sensitivity_results['attribute_specific']['fig_influence'] = fig_influence
                
                # Display detailed table
                st.write("Detailed Attribute Influence:")
                st.table(influence_df.style.format({'Productivity Range': '{:.4f}', 'Influence': '{:.4f}'}))
                
                # Individual attribute details
                for attr, data in results.items():
                    if attr not in ['influence_df', 'fig_influence']:
                        with st.expander(f"Sensitivity Analysis for {attr}", expanded=False):
                            if all(key in data for key in ['attr_min', 'attr_max', 'prod_min', 'prod_max', 'prod_range']):
                                st.write(f"**Attribute Range:** {data['attr_min']:.4f} to {data['attr_max']:.4f}")
                                st.write(f"**Productivity Range:** {data['prod_min']:.4f} to {data['prod_max']:.4f}")
                                st.write(f"**Productivity Spread:** {data['prod_range']:.4f}")
                            else:
                                st.warning(f"Complete data not available for {attr}. Please rerun the analysis.")
                            
                            st.plotly_chart(data['fig'], use_container_width=True)
                            
                            if len(selected_attributes) == 1:
                                st.dataframe(data['results'], use_container_width=True, hide_index=True)
        else:
            st.warning("Please run the General Sensitivity Analysis first.")

    if st.session_state.sensitivity_results['general'] and st.session_state.sensitivity_results['attribute_specific']:
        st.markdown("---")  # Add a horizontal line for separation
        st.subheader("Export Report")
        
        report_format = st.selectbox("Select Report Format", ["PDF", "PowerPoint"], key="sensitivity_report_format")

        if st.button("Generate and Download Report", key="sensitivity_generate_report"):
            try:
                with st.spinner(f"Generating {report_format} report... This may take a few moments."):
                    if report_format == "PDF":
                        report_buffer = generate_pdf_report(
                            equation_to_use,
                            st.session_state.sensitivity_results['general']['baseline_productivity'],
                            st.session_state.sensitivity_results['general']['sensitivity_df'],
                            st.session_state.sensitivity_results['general'],
                            st.session_state.sensitivity_results['attribute_specific'],
                            st.session_state.user_id
                        )
                        file_extension = "pdf"
                        mime_type = "application/pdf"
                    else:  # PowerPoint
                        report_buffer = generate_ppt_report(
                            equation_to_use,
                            st.session_state.sensitivity_results['general']['baseline_productivity'],
                            st.session_state.sensitivity_results['general'],
                            st.session_state.sensitivity_results['attribute_specific'],
                            st.session_state.user_id
                        )
                        file_extension = "pptx"
                        mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                
                if report_buffer:
                    st.download_button(
                        label=f"Download {report_format} Report",
                        data=report_buffer,
                        file_name=f"sensitivity_analysis_report.{file_extension}",
                        mime=mime_type
                    )
                    st.success(f"{report_format} report generated successfully. Click the download button to save it.")
                else:
                    st.error(f"Failed to generate the {report_format} report. Please check the logs for more information.")
            except Exception as e:
                st.error(f"An error occurred while generating the {report_format} report: {str(e)}")
                log_message(logging.ERROR, f"Error in {report_format} report generation: {str(e)}", exc_info=True)
    


# Genetic Algorithm Optimization
def start_ga_optimization(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, excluded_rows, regression_type, num_models):
    full_zscored_df = df.copy()
    df = df[~df['data_id'].isin(excluded_rows)]

    #add predictors to session state variable for model saving
    st.session_state.ga_optimizer['predictors'] = predictors
    
    start_time = time.time()
    timer_placeholder = st.empty()
    plot_placeholder = st.empty()

    # Initialize continuous optimization tracking variables
    if 'continuous_optimization' not in st.session_state:
        st.session_state.continuous_optimization = {
            'r2_values': [],
            'iterations': [],
            'model_markers': {},
            'current_iteration': 0
        }

    while len(st.session_state.ga_optimizer['results']) < num_models and st.session_state.ga_optimizer['running']:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_ga_optimization(
                    df, target_column, predictors, r2_threshold, coef_range,
                    prob_crossover, prob_mutation, num_generations, population_size,
                    timer_placeholder, regression_type,
                    len(st.session_state.ga_optimizer['results']),
                    st.session_state.continuous_optimization['r2_values'],
                    st.session_state.continuous_optimization['iterations'],
                    st.session_state.continuous_optimization['model_markers'],
                    plot_placeholder,
                    st.session_state.continuous_optimization['current_iteration']
                )
                if w:
                    for warning in w:
                        log_message(logging.WARNING, f"Warning in run_ga_optimization: {warning.message}")

            if result:
                try:
                    best_ind, best_r2_score, response_equation, selected_feature_names, errors_df, full_dataset_r2, last_iteration = result
                    # Update the current_iteration with the value returned from run_ga
                    st.session_state.continuous_optimization['current_iteration'] = last_iteration
                except ValueError as e:
                    log_message(logging.ERROR, f"Error unpacking result: {str(e)}")
                    continue

                try:
                    predicted_values = calculate_predicted_productivity(full_zscored_df, response_equation)
                except Exception as e:
                    log_message(logging.ERROR, f"Error calculating predicted values: {str(e)}")
                    continue

                result_with_predictions = (best_ind, best_r2_score, response_equation, selected_feature_names, errors_df, predicted_values, full_zscored_df, excluded_rows, full_dataset_r2)
                
                st.session_state.ga_optimizer['results'].append(result_with_predictions)
                log_message(logging.INFO, f"Model {len(st.session_state.ga_optimizer['results'])} generated (Weighted R²: {best_r2_score:.4f}, Full Dataset R²: {full_dataset_r2:.4f})")
                st.success(f"Model {len(st.session_state.ga_optimizer['results'])} generated (Weighted R²: {best_r2_score:.4f}, Full Dataset R²: {full_dataset_r2:.4f})")
            else:
                log_message(logging.WARNING, "GA optimization did not produce a valid result. Retrying...")
                st.warning("GA optimization did not produce a valid result. Retrying...")
        except Exception as e:
            log_message(logging.ERROR, f"Error in GA optimization: {str(e)}")

    st.session_state.ga_optimizer['running'] = False
    log_message(logging.DEBUG, "GA optimization completed")
    st.rerun()

if __name__ == "__main__":
    main()