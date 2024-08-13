import streamlit as st
import pandas as pd
import numpy as np
from utils.plotting import create_multi_axis_plot, plot_rolling_ir  # Assume plot_rolling_ir is implemented in utils/plotting.py
import logging
from utils.realtime_analytics_db import get_wells, get_stages_for_well, get_well_completion_data, fetch_data_for_modeling
from utils.ga_utils import calculate_df_statistics
from utils.db import fetch_all_models_with_users, fetch_model_details
from logging.handlers import RotatingFileHandler
import os
import traceback
from realtime_analytics.analysis_utils import analyze_data, calculate_rolling_ir, export_to_excel

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
    logger.log(level,f"[realtime-analytics] {message}")


def identify_depletion_region(well_name, well_id, selected_stages):
    st.subheader("Depletion Region Identification")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    plot_placeholder = st.empty()
    debug_placeholder = st.empty()

    all_stage_data = []
    total_stages = len(selected_stages)
    
    log_message(logging.INFO, f"Starting depletion region identification for {well_name} with {total_stages} stages")

    for i, stage in enumerate(selected_stages):
        status_text.text(f"Processing stage {stage} ({i+1}/{total_stages})")
        try:
            log_message(logging.INFO, f"Fetching data for {well_name}, Stage {stage}")
            completion_data = get_well_completion_data(well_id, stage)
            
            if not completion_data:
                log_message(logging.WARNING, f"No data available for {well_name}, Stage {stage}")
                st.warning(f"No data available for {well_name}, Stage {stage}.")
                continue
            
            df = pd.DataFrame(completion_data)
            df = df.sort_values('time_seconds')
            
            log_message(logging.INFO, f"Calculating rolling IR for {well_name}, Stage {stage}")
            rolling_ir_data = calculate_rolling_ir(df, stage)
            
            if rolling_ir_data.empty:
                log_message(logging.WARNING, f"No valid data for rolling IR calculation in {well_name}, Stage {stage}")
                st.warning(f"No valid data for rolling IR calculation in {well_name}, Stage {stage}.")
                continue
            
            rolling_ir_data['stage'] = stage
            all_stage_data.append(rolling_ir_data)
            
            # Debug information
            debug_info = f"Processed Stage {stage}: {len(rolling_ir_data)} data points"
            log_message(logging.INFO, debug_info)
            debug_placeholder.text(debug_info)
            
        except Exception as e:
            error_msg = f"Error processing {well_name}, Stage {stage}: {str(e)}"
            log_message(logging.ERROR, error_msg)
            log_message(logging.ERROR, f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
        
        # Update progress
        progress_bar.progress((i + 1) / total_stages)
        
        # Plot every 5 stages or on the last stage
        if (i + 1) % 5 == 0 or i == total_stages - 1:
            status_text.text(f"Plotting stages 1-{i+1}...")
            try:
                combined_data = pd.concat(all_stage_data, ignore_index=True)
                log_message(logging.INFO, f"Plotting {len(all_stage_data)} stages, total data points: {len(combined_data)}")
                fig = plot_rolling_ir(combined_data, well_name)
                plot_placeholder.plotly_chart(fig, use_container_width=True)
            except Exception as plot_error:
                error_msg = f"Error plotting stages 1-{i+1}: {str(plot_error)}"
                log_message(logging.ERROR, error_msg)
                log_message(logging.ERROR, f"Traceback: {traceback.format_exc()}")
                st.error(error_msg)

    status_text.text("Processing complete!")

    if all_stage_data:
        combined_data = pd.concat(all_stage_data, ignore_index=True)
        st.write(f"Total data points: {len(combined_data)}")
        st.write(f"Unique stages: {combined_data['stage'].nunique()}")
        log_message(logging.INFO, f"Final plot: {combined_data['stage'].nunique()} stages, {len(combined_data)} data points")
        
        try:
            fig = plot_rolling_ir(combined_data, well_name)
            plot_placeholder.plotly_chart(fig, use_container_width=True, height=fig.layout.height)
            
            # Calculate descriptive statistics per stage
            stats_df = combined_data.groupby('stage')['rolling_IR'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max',
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75)
            ]).reset_index()
            stats_df.columns = ['Stage', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile']
            
            # Display the statistics table
            st.subheader("Descriptive Statistics of Rolling IR by Stage")
            st.dataframe(stats_df.style.format({
                'Mean': '{:.5f}',
                'Median': '{:.5f}',
                'Std Dev': '{:.5f}',
                'Min': '{:.5f}',
                'Max': '{:.5f}',
                '25th Percentile': '{:.5f}',
                '75th Percentile': '{:.5f}'
            }), hide_index=True, use_container_width=True)
            
            # Create Excel file
            excel_file = export_to_excel(combined_data, well_name)
            
            # Provide download button
            st.download_button(
                label="Download Rolling IR Data (Excel)",
                data=excel_file,
                file_name=f"{well_name}_rolling_ir_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as final_plot_error:
            error_msg = f"Error creating final plot, statistics table, or Excel file: {str(final_plot_error)}"
            log_message(logging.ERROR, error_msg)
            log_message(logging.ERROR, f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
    else:
        log_message(logging.WARNING, "No valid data available for the selected stages")
        st.warning("No valid data available for the selected stages.")

def display_model_details(model_details):
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Weighted R2 Score", f"{model_details.get('weighted_r2_score', 'N/A'):.4f}")
    
    with col2:
        st.metric("Full Dataset R2 Score", f"{model_details.get('full_dataset_r2_score', 'N/A'):.4f}")

    st.write("**Model Equation:**")
    st.code(model_details.get('equation', 'Equation not available'), language="python")

    st.write("**Predictors:**")
    predictors = model_details.get('predictors', [])
    predictor_html = " ".join([f'<span style="background-color: #007bff; color: white; padding: 2px 6px; margin: 2px; border-radius: 10px;">{pred}</span>' for pred in predictors])
    st.markdown(predictor_html, unsafe_allow_html=True)

    st.write("**Selected Features:**")
    features = model_details.get('selected_feature_names', [])
    feature_html = " ".join([f'<span style="background-color: #28a745; color: white; padding: 2px 6px; margin: 2px; border-radius: 10px;">{feat}</span>' for feat in features])
    st.markdown(feature_html, unsafe_allow_html=True)

def test_saved_models(well_name, well_id, selected_stage):
    st.subheader("Test Saved Models")

    # Fetch all saved models
    saved_models = fetch_all_models_with_users()

    if not saved_models:
        st.warning("No saved models found. Please create and save a model first.")
        return

    # Create a selectbox for model selection
    model_options = [f"{model['model_name']} (created on {model['created_on']})" for model in saved_models]
    selected_model_option = st.selectbox("Select a model", options=model_options)

    if selected_model_option:
        # Extract model_id from the selected option
        selected_model = next((model for model in saved_models if f"{model['model_name']} (created on {model['created_on']})" == selected_model_option), None)
        
        if selected_model:
            # Fetch model details
            model_details = fetch_model_details(selected_model['model_id'])

            if model_details:
                st.write(f"**Created by:** {selected_model['created_by']}")
                
                # Display basic model details
                display_model_details(model_details)

                # Fetch data for modeling and calculate statistics
                data_for_modeling = pd.DataFrame(fetch_data_for_modeling(well_id))
                df_statistics = calculate_df_statistics(data_for_modeling)
                
                st.subheader("Well Statistics")
                st.dataframe(pd.DataFrame(df_statistics))


def main(authentication_status):
    if not authentication_status:
        st.warning("Please login to access this page.")
        return

    try:
        st.title('Real-time Well Completion Data Visualization and Analysis')
        
        # Fetch all wells
        wells = get_wells()
        well_options = {well['well_name']: well['well_id'] for well in wells}
        
        # Create two columns for well and stage selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_well = st.selectbox(
                "Select a Well",
                options=list(well_options.keys()),
                help="Choose the well you want to analyze."
            )
        
        well_id = well_options[selected_well]
        
        with col2:
            stages = get_stages_for_well(well_id)
            
        # Operation selection
        operation = st.selectbox(
            "Select Operation",
            ["Select an operation", "Identify Depletion Region", "Test Saved Models"],
            index=0,
            help="Choose the type of analysis or operation you want to perform."
        )
        
        # Perform the selected operation
        if operation == "Identify Depletion Region":
            selected_stages = st.multiselect(
                "Select Stages",
                options=stages,
                default=stages,
                help="Choose one or more stages to analyze. You can select multiple stages."
            )
            if st.button("Start Depletion Region Identification"):
                if selected_stages:
                    identify_depletion_region(selected_well, well_id, selected_stages)
                else:
                    st.warning("Please select at least one stage before starting the identification process.")
        elif operation == "Test Saved Models":
            selected_stage = st.selectbox(
                "Select Stage",
                options=stages,
                help="Choose the stage of the well you want to analyze."
            )
            test_saved_models(selected_well, well_id, selected_stage)
        elif operation == "Select an operation":
            st.info("Please select an operation to proceed.")
    
    except Exception as e:
        log_message(logging.ERROR, f"Unhandled error in main function: {str(e)}")
        st.error("An error occurred. Please check the logs for more information.")