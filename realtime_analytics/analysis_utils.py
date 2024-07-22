
import numpy as np
import logging
import pandas as pd


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

