import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from utils.db import get_well_details, get_modeling_data, get_well_stages, get_array_data, store_ga_model
import os
from logging.handlers import RotatingFileHandler
import logging
from ga.ga_calculation import run_ga as run_ga_optimization
from ga.check_monotonicity import check_monotonicity as check_monotonicity_func
from utils.plotting import plot_column, plot_actual_vs_predicted
from utils.ga_utils import zscore_data, calculate_df_statistics, validate_custom_equation, calculate_predicted_productivity
import time

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
            'show_zscore_tab': False,
            'regression_type': 'FPR',
            'monotonicity_results': {},
            'df_statistics': None,
            'show_monotonicity': False,
            'selected_wells': [],
            'r2_threshold': 0.55,
            'prob_crossover': 0.8,
            'prob_mutation': 0.2,
            'num_generations': 40,
            'population_size': 50
        }

def main(authentication_status):
    if not authentication_status:
        st.warning("Please login to access this page.")
        log_message(logging.WARNING, "User not authenticated")
        return

    initialize_ga_state()

    st.sidebar.title("GA Optimizer Menu")
    if st.sidebar.button("Monotonicity Check", key="fab_monotonicity"):
        st.session_state.ga_optimizer['show_monotonicity'] = not st.session_state.ga_optimizer['show_monotonicity']

    st.title("Hydraulic Fracturing Productivity Model Optimizer (HF-PMO)")

    ga_optimization_section()

    if st.session_state.ga_optimizer['show_monotonicity']:
        monotonicity_check_modal()

def ga_optimization_section():
    wells = get_well_details()
    well_options = {well['well_name']: well['well_id'] for well in wells}
    
    selected_wells = st.multiselect("Select Wells", options=list(well_options.keys()))
    selected_well_ids = [well_options[well] for well in selected_wells]
    
    st.session_state.ga_optimizer['selected_wells'] = selected_well_ids

    if not selected_wells:
        st.warning("Please select at least one well to proceed.")
        return

    consolidated_data = fetch_consolidated_data(selected_well_ids)
    
    if consolidated_data.empty:
        st.error("No data available for the selected wells.")
        return

    df = consolidated_data.sort_values(by=['Well Name', 'stage'])
    df['Productivity'] = ""

    if st.session_state.ga_optimizer['show_zscore_tab']:
        tab1, tab2 = st.tabs(["Data Preview", "Z-Score Data"])
    else:
        tab1, = st.tabs(["Data Preview"])

    with tab1:
        st.write("Data Preview (You can edit the Productivity column):")
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column("Productivity", editable=True)
        gb.configure_column("Well Name", hide=False)
        for col in df.columns:
            if col not in ['Productivity', 'Well Name']:
                gb.configure_column(col, editable=False)
        gb.configure_grid_options(domLayout='normal', suppressMovableColumns=True, enableRangeSelection=True, clipboardDelimiter=',')
        grid_options = gb.build()
        grid_response = AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.VALUE_CHANGED,
                               fit_columns_on_grid_load=True, height=400, allow_unsafe_jscode=True)
        edited_df = pd.DataFrame(grid_response['data'])
        st.session_state.ga_optimizer['edited_df'] = edited_df
        st.session_state.ga_optimizer['df_statistics'] = calculate_df_statistics(edited_df)

        if st.button("Z-Score Data"):
            if edited_df['Productivity'].isnull().any() or (edited_df['Productivity'] == "").any():
                st.error("Please ensure all values in the 'Productivity' column are filled.")
            else:
                zscored_df = zscore_data(edited_df)
                st.session_state.ga_optimizer['zscored_df'] = zscored_df
                st.session_state.ga_optimizer['show_zscore_tab'] = True
                st.success("Data has been Z-Scored.")
                st.rerun()

    if st.session_state.ga_optimizer['show_zscore_tab']:
        with tab2:
            st.write("Z-Scored Data Preview:")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.ga_optimizer['zscored_df'])
            gb.configure_selection('multiple', use_checkbox=True)
            gb.configure_column("Well Name", hide=False)
            grid_options = gb.build()
            grid_response = AgGrid(st.session_state.ga_optimizer['zscored_df'], gridOptions=grid_options,
                                   update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True,
                                   height=400, allow_unsafe_jscode=True)
            selected_rows = pd.DataFrame(grid_response['selected_rows'])
            st.session_state.ga_optimizer['excluded_rows'] = selected_rows.index.tolist()
            st.write("Selected rows to exclude from GA optimization:")
            st.dataframe(selected_rows, use_container_width=True, hide_index=True)

    with st.expander("Feature Selection"):
        drop_columns = st.multiselect("Select Columns to Drop",
                                      [col for col in df.columns if col not in ['Productivity', 'stage']],
                                      help="Choose columns that you do not want to include in the optimization process.")
        predictors = [col for col in df.columns if col not in ['Productivity', 'stage', 'Well Name'] and col not in drop_columns]

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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start GA Optimization", key="start_button", disabled=st.session_state.ga_optimizer['running']):
            with st.spinner('Running Genetic Algorithm...'):
                st.session_state.ga_optimizer['running'] = True
                st.session_state.ga_optimizer['results'] = []  # Reset results
                st.rerun()

    with col2:
        if st.button("Stop GA Optimization", key="stop_button", disabled=not st.session_state.ga_optimizer['running']):
            st.session_state.ga_optimizer['running'] = False
            st.rerun()

    if st.session_state.ga_optimizer['running']:
        start_ga_optimization(st.session_state.ga_optimizer['zscored_df'], 'Productivity', predictors, st.session_state.ga_optimizer['r2_threshold'],
                              coef_range, st.session_state.ga_optimizer['prob_crossover'], st.session_state.ga_optimizer['prob_mutation'], 
                              st.session_state.ga_optimizer['num_generations'], st.session_state.ga_optimizer['population_size'],
                              st.session_state.ga_optimizer['excluded_rows'],
                              st.session_state.ga_optimizer['regression_type'], num_models)

    if st.session_state.ga_optimizer['results']:
        display_ga_results()

def fetch_consolidated_data(well_ids):
    consolidated_data = pd.DataFrame()
    for well_id in well_ids:
        data = get_modeling_data(well_id)
        df = pd.DataFrame(data)
        if not df.empty:
            well_name = next(well['well_name'] for well in get_well_details() if well['well_id'] == well_id)
            df['Well Name'] = well_name
            consolidated_data = pd.concat([consolidated_data, df], ignore_index=True)
    return consolidated_data

def display_ga_results():
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
            
            st.write("Error Table for Individual Data Points")
            st.dataframe(errors_df, use_container_width=True, hide_index=True)

            st.write("Actual vs Predicted Productivity Plot")
            fig = plot_actual_vs_predicted(zscored_df['Productivity'], predicted_values, excluded_rows, zscored_df['Well Name'], zscored_df['stage'])
            st.plotly_chart(fig, use_container_width=True)

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

def save_model(model_index):
    if 'results' not in st.session_state.ga_optimizer or not st.session_state.ga_optimizer['results']:
        st.error("No models available to save. Please run GA optimization first.")
        return

    if model_index < 0 or model_index >= len(st.session_state.ga_optimizer['results']):
        st.error("Invalid model index. Please select a valid model.")
        return

    model_result = st.session_state.ga_optimizer['results'][model_index]
    best_ind, weighted_r2_score, response_equation, selected_feature_names, errors_df, predicted_values, zscored_df, excluded_rows, full_dataset_r2 = model_result

    model_name = st.text_input(f"Enter name for Model {model_index + 1}:", key=f"model_name_input_{model_index}")
    
    if not model_name:
        st.warning("Please enter a name for the model before saving.")
        return

    try:
        user_id = st.session_state.get('user_id')
        r2_threshold = st.session_state.ga_optimizer.get('r2_threshold')
        prob_crossover = st.session_state.ga_optimizer.get('prob_crossover')
        prob_mutation = st.session_state.ga_optimizer.get('prob_mutation')
        num_generations = st.session_state.ga_optimizer.get('num_generations')
        population_size = st.session_state.ga_optimizer.get('population_size')
        regression_type = st.session_state.ga_optimizer.get('regression_type')
        
        well_ids = st.session_state.ga_optimizer.get('selected_wells', [])
        excluded_stages = [excluded_rows]  # Assuming excluded_rows is a list of stages for all wells

        if not all([user_id, r2_threshold, prob_crossover, prob_mutation, num_generations, population_size, regression_type, well_ids]):
            raise ValueError("One or more required parameters are missing.")

        with st.spinner("Saving model..."):
            model_id = store_ga_model(
                user_id, model_name, weighted_r2_score, full_dataset_r2, response_equation,
                r2_threshold, prob_crossover, prob_mutation, num_generations, population_size,
                regression_type, selected_feature_names, well_ids, excluded_stages
            )
        
        if model_id:
            st.success(f"Model '{model_name}' saved successfully with ID: {model_id}")
            log_message(logging.INFO, f"Model saved successfully. Name: {model_name}, ID: {model_id}")
        else:
            st.error("Failed to save the model. The database operation did not return a model ID.")
            log_message(logging.ERROR, f"Failed to save model '{model_name}'. No model ID returned.")

    except ValueError as ve:
        st.error(f"Error saving model: {str(ve)}")
        log_message(logging.ERROR, f"Error saving model '{model_name}': {str(ve)}")
    except Exception as e:
        st.error("An unexpected error occurred while saving the model. Please try again or contact support.")
        log_message(logging.ERROR, f"Unexpected error saving model '{model_name}': {str(e)}", exc_info=True)

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
    selected_stages = st.multiselect("Select Stage(s)", options=stages, key="monotonicity_stage_select")

    if 'results' in st.session_state.ga_optimizer and st.session_state.ga_optimizer['results']:
        model_options = [f"Model {i+1} (R²: {result[1]:.4f})" for i, result in enumerate(st.session_state.ga_optimizer['results'])]
        model_options.append("Custom Equation")
        selected_model = st.selectbox("Select Model for Monotonicity Check", options=model_options)
    else:
        selected_model = "Custom Equation"
        st.info("No GA models available. Using custom equation.")

    if selected_model == "Custom Equation":
        custom_equation = st.text_area(
            "Enter custom equation", 
            placeholder="Corrected_Prod = ... (use only tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry)",
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
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Monotonicity Check"):
            if equation_to_use is None:
                st.error("Please provide a valid equation before checking monotonicity.")
            elif not selected_stages:
                st.error("Please select at least one stage for monotonicity check.")
            else:
                run_monotonicity_check(well_id, selected_stages, equation_to_use)
    
    with col2:
        if st.button("Save Model"):
            if selected_model != "Custom Equation":
                model_index = int(selected_model.split()[1]) - 1
                save_model(model_index)
            else:
                st.warning("Cannot save a custom equation. Please run GA optimization to generate a model first.")

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
            st.subheader("Productivity Plots for Selected Stages")
            for stage, result_df in st.session_state.ga_optimizer['monotonicity_results'].items():
                fig = plot_column(result_df, 'Productivity', stage)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Monotonicity check to see results.")

def start_ga_optimization(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, excluded_rows, regression_type, num_models):
    full_zscored_df = df.copy()
    df = df.drop(excluded_rows)
    
    start_time = time.time()
    timer_placeholder = st.empty()

    while len(st.session_state.ga_optimizer['results']) < num_models and st.session_state.ga_optimizer['running']:
        result = run_ga_optimization(
            df, target_column, predictors, r2_threshold, coef_range,
            prob_crossover, prob_mutation, num_generations, population_size,
            timer_placeholder, regression_type
        )

        if result:
            best_ind, best_r2_score, response_equation, selected_feature_names, errors_df, full_dataset_r2 = result
            
            predicted_values = calculate_predicted_productivity(full_zscored_df, response_equation)
            
            result_with_predictions = (best_ind, best_r2_score, response_equation, selected_feature_names, errors_df, predicted_values, full_zscored_df, excluded_rows, full_dataset_r2)
            
            st.session_state.ga_optimizer['results'].append(result_with_predictions)
            log_message(logging.INFO, f"Model generated (Weighted R²: {best_r2_score:.4f}, Full Dataset R²: {full_dataset_r2:.4f})")
            st.success(f"Model {len(st.session_state.ga_optimizer['results'])} generated (Weighted R²: {best_r2_score:.4f}, Full Dataset R²: {full_dataset_r2:.4f})")
        else:
            log_message(logging.WARNING, "GA optimization did not produce a valid result. Retrying...")
            st.warning("GA optimization did not produce a valid result. Retrying...")

    st.session_state.ga_optimizer['running'] = False
    log_message(logging.DEBUG, "GA optimization completed")
    st.rerun()

if __name__ == "__main__":
    main()