import streamlit as st
import pandas as pd
import numpy as np
from utils.plotting import create_multi_axis_plot
import logging
from utils.realtime_analytics_db import get_wells, get_stages_for_well, get_well_completion_data
from logging.handlers import RotatingFileHandler
import os

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

def calculate_slope(x, y):
    try:
        return np.polyfit(x, y, 1)[0]
    except Exception as e:
        log_message(logging.ERROR, f"Error calculating slope: {str(e)}")
        raise

def analyze_window(window):
    try:
        time = window['Time Seconds']
        tp = window['Treating Pressure']
        sr = window['Slurry Rate']
        ppc = window['PPC']
        bh_prop_mass = window['BH Prop Mass']
        
        tp_slope = calculate_slope(time, tp)
        sr_slope = calculate_slope(time, sr)
        ppc_change = ppc.iloc[-1] - ppc.iloc[0]
        
        condition1 = tp_slope < 0 and sr_slope > 0
        condition2 = tp_slope < 0 and abs(sr_slope) < 1e-6
        
        # New analysis
        min_tp_idx = tp.idxmin()
        max_tp_idx = tp.idxmax()
        min_tp = tp.loc[min_tp_idx]
        max_tp = tp.loc[max_tp_idx]
        sr_at_min_tp = sr.loc[min_tp_idx]
        sr_at_max_tp = sr.loc[max_tp_idx]
        bh_at_min_tp = bh_prop_mass.loc[min_tp_idx]
        bh_at_max_tp = bh_prop_mass.loc[max_tp_idx]
        
        new_analysis = {
            'Min Treating Pressure': min_tp,
            'Max Treating Pressure': max_tp,
            'Slurry Rate @ Min TP': sr_at_min_tp,
            'Slurry Rate @ Max TP': sr_at_max_tp,
            'BH Prop Mass @ Min TP': bh_at_min_tp,
            'BH Prop Mass @ Max TP': bh_at_max_tp,
            'Delta Treating Pressure': max_tp - min_tp,
            'Delta Slurry Rate': sr_at_max_tp - sr_at_min_tp,
            'Delta BH Prop Mass': bh_at_max_tp - bh_at_min_tp
        }
        
        return (condition1 or condition2), ppc_change, new_analysis
    except Exception as e:
        log_message(logging.ERROR, f"Error analyzing window: {str(e)}")
        raise

def detect_leakoff_periods(data):
    try:
        leakoff_periods = []
        in_leakoff = False
        start_time = None
        found_first_slurry = False

        for index, row in data.iterrows():
            if row['Slurry Rate'] > 0:
                found_first_slurry = True
            if found_first_slurry:
                if row['Slurry Rate'] == 0 and row['Treating Pressure'] > 0:
                    if not in_leakoff:
                        # Start of a leakoff period
                        in_leakoff = True
                        start_time = row['datetime']
                else:
                    if in_leakoff:
                        # End of a leakoff period
                        in_leakoff = False
                        end_time = row['datetime']
                        leakoff_periods.append((start_time, end_time))
                        start_time = None

        if in_leakoff:
            # If the data ends while still in a leakoff period
            leakoff_periods.append((start_time, data['datetime'].iloc[-1]))

        return leakoff_periods
    except Exception as e:
        log_message(logging.ERROR, f"Error detecting leakoff periods: {str(e)}")
        raise

def analyze_data(df, window_size):
    try:
        total_time = df['Time Seconds'].max() - df['Time Seconds'].min()
        window_step = (window_size + 1) // 2
        total_windows = max(1, int((total_time - window_size) / window_step) + 1)
        
        event_count = 0
        total_ppc_change = 0
        event_windows = []
        new_analysis_results = []
        
        leakoff_periods = detect_leakoff_periods(df)
        
        log_message(logging.INFO, f"Starting analysis with {total_windows} windows")
        
        for i in range(total_windows):
            start_time = df['Time Seconds'].min() + i * window_step
            end_time = start_time + window_size
            window = df[(df['Time Seconds'] >= start_time) & (df['Time Seconds'] < end_time)]
            
            if len(window) > 0:
                event, ppc_change, new_analysis = analyze_window(window)
                new_analysis['Window Start'] = start_time
                new_analysis['Window End'] = end_time
                new_analysis_results.append(new_analysis)
                
                if event:
                    is_in_leakoff = any(start <= window['datetime'].iloc[0] <= end for start, end in leakoff_periods)
                    if not is_in_leakoff:
                        event_count += 1
                        total_ppc_change += ppc_change
                        event_windows.append((start_time, end_time))
            
            if i % 100 == 0:
                log_message(logging.INFO, f"Processed {i}/{total_windows} windows")
        
        log_message(logging.INFO, f"Analysis completed: {event_count} events found out of {total_windows} windows")
        return total_windows, event_count, total_ppc_change, event_windows, leakoff_periods, new_analysis_results
    except Exception as e:
        log_message(logging.ERROR, f"Error analyzing data: {str(e)}")
        raise



def main(authentication_status):
    if not authentication_status:
        st.warning("Please login to access this page.")
        return

    try:
        st.title('Real-time Well Completion Data Visualization and Analysis')
        
        # Fetch all wells
        wells = get_wells()
        well_options = {f"{well['well_name']} (ID: {well['well_id']})": well['well_id'] for well in wells}
        
        # Well selection dropdown
        selected_well = st.selectbox("Select a Well", options=list(well_options.keys()))
        well_id = well_options[selected_well]
        
        # Fetch stages for selected well
        stages = get_stages_for_well(well_id)
        
        # Stage selection dropdown
        selected_stage = st.selectbox("Select a Stage", options=stages)
        
        window_size = st.slider("Select window size (seconds)", min_value=11, max_value=31, value=20, step=1)
        
        if st.button("Start Processing"):
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
            
            plot_title = f"{selected_well} - Stage {selected_stage}"
            
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
                "Well Name": selected_well.split(" (ID:")[0],
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
            
            log_message(logging.INFO, f"Analysis completed for {selected_well} - Stage {selected_stage}")
    
    except Exception as e:
        log_message(logging.ERROR, f"Unhandled error in main function: {str(e)}")
        st.error("An error occurred. Please check the logs for more information.")
    