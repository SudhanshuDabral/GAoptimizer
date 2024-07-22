import streamlit as st
import pandas as pd
import numpy as np
from utils.plotting import create_multi_axis_plot, plot_rolling_ir  # Assume plot_rolling_ir is implemented in utils/plotting.py
import logging
from utils.realtime_analytics_db import get_wells, get_stages_for_well, get_well_completion_data
from logging.handlers import RotatingFileHandler
import os
from realtime_analytics.analysis_utils import analyze_data

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

def window_analysis(well_name, well_id, selected_stage):
    st.subheader("Window Analysis Settings")
    window_size = st.slider("Select window size (seconds)", min_value=11, max_value=31, value=20, step=1)
    
    if st.button("Start Window Analysis"):
        # Fetch well completion data
        completion_data = get_well_completion_data(well_id, selected_stage)
        
        if not completion_data:
            st.warning("No data available for the selected well and stage.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(completion_data)
        df['datetime'] = pd.to_datetime(df['epoch'], unit='ms')
        
        # Rename columns to match the expected format
        df = df.rename(columns={
            'treating_pressure': 'Treating Pressure',
            'slurry_rate': 'Slurry Rate',
            'bottomhole_prop_mass': 'BH Prop Mass'
        })
        
        # Add PPC column (you might need to adjust this calculation based on your data)
        df['PPC'] = df['Treating Pressure'] / df['Slurry Rate']
        
        plot_title = f"{well_name} - Stage {selected_stage}"
        
        st.subheader(plot_title)
        total_windows, event_count, total_ppc_change, event_windows, leakoff_periods, new_analysis_results = analyze_data(df, window_size)
        
        log_message(logging.INFO, f"Creating plot for {plot_title}")
        try:
            fig = create_multi_axis_plot(df, plot_title, event_windows, leakoff_periods)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            log_message(logging.ERROR, f"Error creating plot for {plot_title}: {str(e)}")
            st.error(f"Error creating plot for {plot_title}. Please check the logs for more information.")
        
        # Display analysis results
        st.subheader("Analysis Results")
        analysis_results = {
            "Well Name": well_name,
            "Stage": selected_stage,
            "Total Windows": total_windows,
            "Event Windows": event_count,
            "Total PPC Change": total_ppc_change,
            "Average PPC Change": total_ppc_change / event_count if event_count > 0 else 0
        }
        st.table(pd.DataFrame([analysis_results]))
        
        # Display detailed window analysis results
        st.subheader("Detailed Window Analysis Results")
        new_results_df = pd.DataFrame(new_analysis_results)
        st.dataframe(new_results_df, use_container_width=True, hide_index=True)
        
        log_message(logging.INFO, f"Analysis completed for {well_name} - Stage {selected_stage}")

def calculate_rolling_ir(df, stage):
    log_message(logging.INFO, f"Starting rolling IR calculation for stage {stage}")
    log_message(logging.INFO, f"DataFrame shape: {df.shape}")
    log_message(logging.INFO, f"Columns: {df.columns}")
    
    # Check if required columns exist
    required_columns = ['time_seconds', 'slurry_rate', 'treating_pressure']
    for col in required_columns:
        if col not in df.columns:
            log_message(logging.ERROR, f"Required column '{col}' not found in DataFrame for stage {stage}")
            return pd.DataFrame(columns=['time_seconds', 'rolling_IR'])

    # Filter data for the first 10 minutes (600 seconds)
    start_time = df['time_seconds'].min()
    df_filtered = df[(df['time_seconds'] >= start_time) & (df['time_seconds'] <= start_time + 600)]
    
    # Find the index where slurry_rate first exceeds 5 within the first 10 minutes
    start_indices = df_filtered[df_filtered['slurry_rate'] > 5].index
    log_message(logging.INFO, f"Number of indices where slurry_rate > 5 in first 10 minutes: {len(start_indices)}")
    
    if len(start_indices) == 0:
        log_message(logging.WARNING, f"No data points where slurry_rate > 5 in first 10 minutes for stage {stage}")
        return pd.DataFrame(columns=['time_seconds', 'rolling_IR'])
    
    start_index = start_indices[0]
    log_message(logging.INFO, f"Start index: {start_index}")
    
    # Calculate cumulative sums from the start_index
    df_filtered['cum_slurry_rate'] = df_filtered.loc[start_index:, 'slurry_rate'].cumsum()
    df_filtered['cum_treating_pressure'] = df_filtered.loc[start_index:, 'treating_pressure'].cumsum()
    
    # Calculate rolling_IR (slope)
    df_filtered['rolling_IR'] = df_filtered['cum_slurry_rate'] / df_filtered['cum_treating_pressure']
    
    # Normalize time_seconds to start from 0 for each stage
    df_filtered['normalized_time'] = df_filtered['time_seconds'] - df_filtered['time_seconds'].min()
    
    log_message(logging.INFO, f"Filtered DataFrame shape: {df_filtered.shape}")
    
    return df_filtered[['normalized_time', 'rolling_IR']]

def identify_depletion_region(well_name, well_id, selected_stages):
    st.subheader("Depletion Region Identification")
    
    all_stage_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, stage in enumerate(selected_stages):
        status_text.text(f"Processing stage {stage}...")
        try:
            completion_data = get_well_completion_data(well_id, stage)
            
            if not completion_data:
                st.warning(f"No data available for {well_name}, Stage {stage}.")
                continue
            
            df = pd.DataFrame(completion_data)
            df = df.sort_values('time_seconds')
            
            rolling_ir_data = calculate_rolling_ir(df, stage)
            
            if rolling_ir_data.empty:
                st.warning(f"No valid data for rolling IR calculation in {well_name}, Stage {stage}.")
                continue
            
            rolling_ir_data['stage'] = stage
            all_stage_data.append(rolling_ir_data)
        except Exception as e:
            st.error(f"Error processing {well_name}, Stage {stage}: {str(e)}")
            log_message(logging.ERROR, f"Error processing {well_name}, Stage {stage}: {str(e)}")
        finally:
            progress_bar.progress((i + 1) / len(selected_stages))

    status_text.text("Processing complete!")

    if all_stage_data:
        combined_data = pd.concat(all_stage_data)
        
        # Print combined data
        st.subheader("Combined Data Preview")
        st.write("First few rows of the combined data:")
        st.write(combined_data.head())
        
        st.write("Data shape:", combined_data.shape)
        st.write("Columns:", combined_data.columns.tolist())
        
        st.write("Summary statistics:")
        st.write(combined_data.describe())
        
        st.write("Data types:")
        st.write(combined_data.dtypes)
        
        # Log the entire combined data for debugging
        log_message(logging.DEBUG, f"Combined data:\n{combined_data.to_string()}")
        
        fig = plot_rolling_ir(combined_data, well_name)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid data available for the selected stages.")


def test_saved_models(well_name, well_id, selected_stages):
    st.info("Test Saved Models functionality will be implemented here.")
    st.write(f"This feature will allow you to test and evaluate saved models on well data for {well_name}, Stages {', '.join(map(str, selected_stages))}.")

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
            selected_stages = st.multiselect(
                "Select Stage(s)",
                options=stages,
                default=[stages[0]] if stages else [],
                help="Choose one or more stages of the well you want to analyze."
            )
        
        # Operation selection
        operation = st.selectbox(
            "Select Operation",
            ["Window Analysis", "Identify Depletion Region", "Test Saved Models"],
            help="Choose the type of analysis or operation you want to perform."
        )
        
        # Perform the selected operation
        if operation == "Window Analysis":
            if len(selected_stages) == 1:
                window_analysis(selected_well, well_id, selected_stages[0])
            else:
                st.warning("Please select a single stage for Window Analysis.")
        elif operation == "Identify Depletion Region":
            if st.button("Start Depletion Region Identification"):
                identify_depletion_region(selected_well, well_id, selected_stages)
        elif operation == "Test Saved Models":
            test_saved_models(selected_well, well_id, selected_stages)
    
    except Exception as e:
        log_message(logging.ERROR, f"Unhandled error in main function: {str(e)}")
        st.error("An error occurred. Please check the logs for more information.")