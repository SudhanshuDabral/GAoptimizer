import json
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import logging
from scipy import signal
from scipy.ndimage import median_filter
from utils.db import (update_modeling_data, call_insert_arrays_data, 
                      create_new_well, get_well_details, bulk_insert_well_completion_records, log_message)

def apply_butterworth_denoising(pressure_data, window_size=11, hop_size=5, order=1, cutoff=0.5):
    """
    Apply Butterworth filter denoising with windowed approach, mimicking MATLAB implementation.
    
    Parameters:
    - pressure_data: numpy array of pressure values
    - window_size: window size for filtering (should be odd)
    - hop_size: hop size for overlapping windows
    - order: Butterworth filter order
    - cutoff: normalized cutoff frequency (0 < cutoff < 1) - MATLAB Wn parameter
    
    Returns:
    - denoised_pressure: filtered pressure data
    - signal_denoised: boolean flag indicating if filtering was applied
    """
    n = len(pressure_data)
    signal_denoised = False
    
    try:
        # MATLAB: [b,a] = butter(order, Wn, 'low') where Wn = 0.5
        # In MATLAB, Wn is already normalized (0 < Wn < 1)
        # In scipy, we need to ensure the cutoff is valid
        if not (0 < cutoff < 1):
            raise ValueError(f"Cutoff frequency must be between 0 and 1, got {cutoff}")
            
        b, a = signal.butter(order, cutoff, btype='low', analog=False)
        
        denoised_sum = np.zeros(n)
        denoised_weight = np.zeros(n)
        
        # Apply windowed filtering with overlap (following MATLAB logic exactly)
        # MATLAB: for i = 1:hop:(N - WS + 1)
        for i in range(0, n - window_size + 1, hop_size):
            # MATLAB: idxw = i:(i + WS - 1)
            idx_start = i
            idx_end = i + window_size
            window_data = pressure_data[idx_start:idx_end]
            
            # MATLAB: if numel(win) >= 3
            if len(window_data) >= 3:
                # MATLAB: filtered = filtfilt(b, a, win)
                filtered_window = signal.filtfilt(b, a, window_data)
            else:
                filtered_window = window_data  # too short to filter
            
            # MATLAB: denoised_sum(idxw) = denoised_sum(idxw) + filtered(:)
            denoised_sum[idx_start:idx_end] += filtered_window
            denoised_weight[idx_start:idx_end] += 1
        
        # MATLAB: denoised_Pres = denoised_sum ./ max(denoised_weight,1)
        denoised_pressure = denoised_sum / np.maximum(denoised_weight, 1)
        
        # MATLAB: Global fallback where weight==0
        zero_weight_mask = denoised_weight == 0
        if np.any(zero_weight_mask):
            # MATLAB: if numel(Pres) >= 3, global_filtered = filtfilt(b,a,Pres)
            if len(pressure_data) >= 3:
                global_filtered = signal.filtfilt(b, a, pressure_data)
            else:
                global_filtered = pressure_data
            denoised_pressure[zero_weight_mask] = global_filtered[zero_weight_mask]
        
        signal_denoised = True
        
    except Exception as e:
        # MATLAB fallback: denoised_Pres = movmedian(Pres, WS, 'omitnan')
        # Use median filter as fallback (equivalent to MATLAB's movmedian)
        try:
            denoised_pressure = median_filter(pressure_data.astype(float), size=window_size, mode='reflect')
        except:
            # Ultimate fallback - return original data
            denoised_pressure = pressure_data.copy()
        signal_denoised = False
    
    return denoised_pressure, signal_denoised

def initialize_data_prep_state():
    if 'data_prep' not in st.session_state:
        st.session_state.data_prep = {
            'well_details': None,
            'well_id': None,
            'processing_files': False,
            'consolidated_output': None
        }

def main(authentication_status):
    if not authentication_status:
        st.warning("Please login to access this page.")
        return

    initialize_data_prep_state()

    st.title("Data Preparation")
    st.write("Select an existing well or create a new one, then upload CSV files for each stage of the well completion.")

    # Get existing wells
    existing_wells = get_well_details()
    
    # Well selection or creation
    well_option = st.radio("Choose an option:", ["Select Existing Well", "Create New Well"])

    if well_option == "Select Existing Well":
        if existing_wells:
            selected_well = st.selectbox("Select a well:", 
                                         options=[f"{well['well_name']} (API: {well['well_api']})" for well in existing_wells],
                                         format_func=lambda x: x.split(" (API:")[0])
            if selected_well:
                well_name = selected_well.split(" (API:")[0]
                selected_well_details = next((well for well in existing_wells if well['well_name'] == well_name), None)
                if selected_well_details:
                    st.session_state.data_prep['well_id'] = selected_well_details['well_id']
                    st.session_state.data_prep['well_details'] = {
                        "Well Name": selected_well_details['well_name'],
                        "Well API": selected_well_details['well_api'],
                        "Latitude": selected_well_details['latitude'],
                        "Longitude": selected_well_details['longitude'],
                        "TVD (ft)": selected_well_details['tvd'],
                        "Reservoir Pressure": selected_well_details['reservoir_pressure']
                    }
        else:
            st.warning("No existing wells found. Please create a new well.")
            well_option = "Create New Well"

    if well_option == "Create New Well":
        with st.form("well_details_form"):
            well_name = st.text_input("Well Name")
            well_api = st.text_input("Well API")
            latitude = st.text_input("Latitude")
            longitude = st.text_input("Longitude")
            tvd = st.text_input("TVD (ft)")
            reservoir_pressure = st.text_input("Reservoir Pressure")

            submitted = st.form_submit_button("Submit")

            if submitted:
                # Validate Well Name
                if len(well_name) > 255:
                    st.error("Well Name should be less than 255 characters.")
                # Validate Well API
                well_api = re.sub(r'[^0-9]', '', well_api)
                if not well_api.isdigit():
                    st.error("Well API should contain only numbers.")
                # Validate Latitude
                try:
                    latitude = float(latitude)
                    if not (-90 <= latitude <= 90):
                        st.error("Latitude should be between -90 and 90.")
                except ValueError:
                    st.error("Latitude should be a valid number.")
                # Validate Longitude
                try:
                    longitude = float(longitude)
                    if not (-180 <= longitude <= 180):
                        st.error("Longitude should be between -180 and 180.")
                except ValueError:
                    st.error("Longitude should be a valid number.")
                # Validate TVD
                try:
                    tvd = float(tvd)
                    if not (500 <= tvd <= 15500):
                        st.error("TVD should be between 500 ft and 15500 ft.")
                except ValueError:
                    st.error("TVD should be a valid number.")
                # Validate Reservoir Pressure
                try:
                    reservoir_pressure = float(reservoir_pressure)
                    if not (2000 <= reservoir_pressure <= 8500):
                        st.error("Reservoir Pressure should be between 2000 psig and 8500 psig.")
                except ValueError:
                    st.error("Reservoir Pressure should be a valid number.")

                # Check if well already exists
                well_exists = any(
                    well['well_name'] == well_name or well['well_api'] == well_api
                    for well in existing_wells
                )
                if well_exists:
                    st.error("A well with this name or API already exists. Please use a unique name and API.")
                    return

                # If all validations pass, create the new well
                if (
                    len(well_name) <= 255 and well_api.isdigit() and
                    (-90 <= latitude <= 90) and (-180 <= longitude <= 180) and
                    (500 <= tvd <= 15500) and (2000 <= reservoir_pressure <= 8500)
                ):
                    new_well_id = create_new_well(well_name, well_api, latitude, longitude, tvd, reservoir_pressure, st.session_state['user_id'])
                    if new_well_id:
                        st.success("New well created successfully!")
                        st.session_state.data_prep['well_id'] = new_well_id
                        st.session_state.data_prep['well_details'] = {
                            "Well Name": well_name,
                            "Well API": well_api,
                            "Latitude": latitude,
                            "Longitude": longitude,
                            "TVD (ft)": tvd,
                            "Reservoir Pressure": reservoir_pressure
                        }
                        st.rerun()
                    else:
                        st.error("Failed to create new well. Please try again.")

    if st.session_state.data_prep['well_id'] is not None:
        well_details = st.session_state.data_prep['well_details']
        
        if well_option == "Select Existing Well":
            st.subheader("Well Information")
            st.dataframe(pd.DataFrame([well_details]), hide_index=True, use_container_width=True)
            st.write("Well details submitted. Now upload the CSV files for each stage.")

        uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

        if uploaded_files:
            if not st.session_state.data_prep['processing_files']:
                st.session_state.data_prep['processing_files'] = True
                process_files(uploaded_files, well_details)
                st.session_state.data_prep['processing_files'] = False

            # Show the consolidated results in a table
            if st.session_state.data_prep['consolidated_output'] is not None:
                st.subheader("Data for Modelling")
                st.dataframe(st.session_state.data_prep['consolidated_output'], use_container_width=True, hide_index=True)

            # Add a button to save data in DB
            if st.button("Save Data in DB"):
                save_data_in_db(st.session_state.data_prep['well_id'], st.session_state.data_prep['consolidated_output'], st.session_state['user_id'])

def extract_stage(filename):
    # Method 1: Current method (looking for _number.csv)
    match = re.search(r'_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    
    # Method 2: New method (looking for _stage_number)
    match = re.search(r'_stage_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Method 3: Fallback method (looking for any number after an underscore)
    match = re.search(r'_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # If no method works, return None
    st.warning(f"Could not extract stage number from filename: {filename}")
    return None

def process_files(files, well_details):
    all_results = pd.DataFrame()

    user_id = st.session_state.user_id
    user_dir = os.path.join('export', str(user_id))
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, file in enumerate(files):
        # Update progress bar
        progress_bar.progress((i + 1) / len(files))
        progress_text.text(f"Processing file {i + 1} of {len(files)}: {file.name}")

        # Save the uploaded file temporarily
        temp_file_path = os.path.join(user_dir, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

        # Extract the stage from the filename
        stage = extract_stage(file.name)
        if stage is None:
            continue  # Skip this file if the stage value is invalid

        # Load the CSV file using column headers
        try:
            data = pd.read_csv(temp_file_path)
        except Exception as e:
            st.error(f"Error reading CSV file {file.name}: {e}")
            os.remove(temp_file_path)
            continue

        # Create synthetic timestamps if needed
        if 'time' in data.columns and 'epoch' not in data.columns and 'timestamp_utc' not in data.columns:
            # Create a base timestamp (e.g., start of 2024)
            base_timestamp = pd.Timestamp('2024-01-01')
            # Convert time in seconds to timedelta and add to base timestamp
            data['timestamp_utc'] = base_timestamp + pd.to_timedelta(data['time'], unit='s')
            # Create epoch (Unix timestamp in seconds)
            data['epoch'] = data['timestamp_utc'].astype('int64') // 10**9
            T = data['time'].values
        elif 'timestamp_utc' in data.columns:
            # Ensure timestamp_utc is in datetime format and calculate seconds from the first timestamp
            data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'])
            data['time'] = (data['timestamp_utc'] - data['timestamp_utc'].min()).dt.total_seconds()
            # Create epoch (Unix timestamp in seconds)
            data['epoch'] = data['timestamp_utc'].astype('int64') // 10**9
            T = data['time'].values
        elif 'time' in data.columns:
            T = data['time'].values
            # If no epoch column exists, create synthetic timestamps
            if 'epoch' not in data.columns:
                # Create a base timestamp (e.g., start of 2024)
                base_timestamp = pd.Timestamp('2024-01-01')
                # Convert time in seconds to timedelta and add to base timestamp
                data['timestamp_utc'] = base_timestamp + pd.to_timedelta(data['time'], unit='s')
                # Create epoch (Unix timestamp in seconds)
                data['epoch'] = data['timestamp_utc'].astype('int64') // 10**9
        else:
            st.error(f"Required column 'time' or 'timestamp_utc' not found in file: {file.name}")
            os.remove(temp_file_path)
            continue

        # Extract data using column headers (following MATLAB column requirements)
        # MATLAB requires: "treating_pressure_fv","slurry_rate","bottomhole_prop_mass"
        try:
            SlRate = data['slurry_rate'].values
            # Handle different pressure column names
            if 'treating_pressure_fv' in data.columns:
                Pres = data['treating_pressure_fv'].values
            elif 'treating_pressure' in data.columns:
                Pres = data['treating_pressure'].values
            else:
                raise KeyError("Neither 'treating_pressure_fv' nor 'treating_pressure' found")
        except KeyError as e:
            st.error(f"Missing required column in {file.name}: {e}")
            os.remove(temp_file_path)
            continue
        
        # Handle different column names for bottomhole proppant mass
        if 'bottomhole_prop_mass' in data.columns:
            DownHoleProp = data['bottomhole_prop_mass'].values
            prop_mass_col = 'bottomhole_prop_mass'
        elif 'bottom_prop_mass' in data.columns:
            DownHoleProp = data['bottom_prop_mass'].values
            data['bottomhole_prop_mass'] = data['bottom_prop_mass']  # Add standardized column name
            prop_mass_col = 'bottomhole_prop_mass'
        else:
            st.error(f"Required column 'bottomhole_prop_mass' or 'bottom_prop_mass' not found in file: {file.name}")
            os.remove(temp_file_path)
            continue

        # Use the correct pressure column name for database insertion
        pressure_col_name = 'treating_pressure_fv' if 'treating_pressure_fv' in data.columns else 'treating_pressure'
        
        completion_data = pd.DataFrame({
            'treating_pressure': data[pressure_col_name],
            'slurry_rate': data['slurry_rate'],
            'bottomhole_prop_mass': DownHoleProp,
            'time_seconds': T.astype(int),
            'epoch': data['epoch']
        })

        insert_result = bulk_insert_well_completion_records(st.session_state.data_prep['well_id'], stage, completion_data, user_id)

        if insert_result['status'] == 'error':
            st.error(f"Failed to insert well completion records for stage {stage}: {insert_result['message']}")
            continue
        

        # Filter by SlRate > 2 and treating pressure > 1000
        IndexS = np.where((SlRate > 2) & (Pres > 1000))[0]
        T = T[IndexS]
        SlRate = SlRate[IndexS]
        Pres = Pres[IndexS]
        DownHoleProp = DownHoleProp[IndexS]

        n = len(T)
        
        # Apply Butterworth denoising to pressure data (mimicking MATLAB implementation)
        Pres, _ = apply_butterworth_denoising(Pres)

        WS = 11  # Window size, should always be odd
        MP = (WS + 1) // 2

        # Initialize arrays (following MATLAB structure)
        SlpeSR = np.zeros((2, n))  # polyfit results have 2 coefficients
        SumSLR = np.zeros(n)
        WinProp = np.zeros(n)
        Pmaxmin = np.zeros(n)
        SlpeP = np.zeros((2, n))
        PreWinSlurry = np.zeros(n)
        PreWinPmaxmin = np.zeros(n)

        lastValid = 0
        # MATLAB: for i = 1:n (1-based indexing)
        # Python: for i in range(1, n+1) to match MATLAB's 1-based loop
        for i in range(1, n + 1):
            # MATLAB: LI = MP + (i-2)*(MP-1); UL = MP + (i-2)*(MP-1) + WS - 1;
            # Direct translation (MP is already calculated as (WS+1)//2)
            LI = MP + (i - 2) * (MP - 1)
            UL = MP + (i - 2) * (MP - 1) + WS - 1
            
            # MATLAB: if UL > n || LI < 1, break; end
            # Convert to Python 0-based: LI < 1 becomes LI < 0, UL > n stays same
            if UL > n or LI < 1:
                break
            lastValid = i
            
            # Convert MATLAB 1-based indices to Python 0-based for array access
            LI_py = LI - 1  # Convert to 0-based
            UL_py = UL      # UL is exclusive in Python slicing
            
            # MATLAB: xw = (LI:UL)'; yS = SlRate(LI:UL); etc.
            xw = np.arange(LI, UL + 1)  # MATLAB includes both endpoints
            yS = SlRate[LI_py:UL_py]
            yP = Pres[LI_py:UL_py]
            yM = DownHoleProp[LI_py:UL_py]
            
            # MATLAB: SlpeSR(:,i) = polyfit(xw, yS, 1);
            coef_SR = np.polyfit(xw, yS, 1)
            coef_P = np.polyfit(xw, yP, 1)
            SlpeSR[:, i-1] = coef_SR  # Convert i to 0-based for Python array
            SlpeP[:, i-1] = coef_P
            
            # MATLAB: SumSLR(i) = sum(yS)*0.7;
            SumSLR[i-1] = np.sum(yS) * 0.7  # Convert i to 0-based
            WinProp[i-1] = np.sum(yM)
            Pmaxmin[i-1] = np.max(yP) - np.min(yP)
            
            # MATLAB: if i > 1
            if i > 1:
                # MATLAB: PrevLI = LI - WS; PrevUL = LI - 1;
                PrevLI = LI - WS
                PrevUL = LI - 1
                # MATLAB: if PrevLI >= 1
                if PrevLI >= 1:
                    # Convert to Python 0-based indexing
                    PrevLI_py = PrevLI - 1
                    PrevUL_py = PrevUL  # Exclusive end
                    ySprev = SlRate[PrevLI_py:PrevUL_py]
                    yPprev = Pres[PrevLI_py:PrevUL_py]
                    PreWinSlurry[i-1] = np.sum(ySprev) * 0.7
                    PreWinPmaxmin[i-1] = np.max(yPprev) - np.min(yPprev)

        # MATLAB: if lastValid == 0, warning(...); continue; end
        if lastValid == 0:
            st.warning(f'Skipping {file.name}: not enough samples for window size {WS}.')
            os.remove(temp_file_path)
            continue
            
        # MATLAB: rngIdx = 1:lastValid;
        # MATLAB: SumSLR = SumSLR(rngIdx)'; Pmaxmin = Pmaxmin(rngIdx)'; WinProp = WinProp(rngIdx)';
        # Convert to Python 0-based indexing
        rng_idx = slice(0, lastValid)  # lastValid is already 1-based from MATLAB loop
        SumSLR = SumSLR[rng_idx]
        Pmaxmin = Pmaxmin[rng_idx]
        WinProp = WinProp[rng_idx]
        
        # MATLAB: dpdt = SlpeP(1,rngIdx)'; dSdt = SlpeSR(1,rngIdx)';
        dpdt = SlpeP[0, rng_idx]  # MATLAB's 1st row becomes 0th row in Python
        dSdt = SlpeSR[0, rng_idx]
        PreWinSlurry = PreWinSlurry[rng_idx]
        PreWinPmaxmin = PreWinPmaxmin[rng_idx]
        
        # MATLAB: drdp = dSdt ./ dpdt
        # Handle division by zero safely before MATLAB's isfinite check
        with np.errstate(divide='ignore', invalid='ignore'):
            drdp = dSdt / dpdt
        
        # MATLAB: Replace non-finite slopes/ratios
        # dpdt(~isfinite(dpdt)) = 0; drdp(~isfinite(drdp)) = 0;
        dpdt[~np.isfinite(dpdt)] = 0
        drdp[~np.isfinite(drdp)] = 0

        C = len(drdp)
        count = np.zeros(C)
        SlrWin = np.zeros(C)
        PmaxminWin = np.zeros(C)
        DownholeWinProp = np.zeros(C)
        EnergyProxy = np.zeros(C)
        EnergyDissipated = np.zeros(C)
        SumPreWinSlurry = np.zeros(C)
        SumPreWinPmaxmin = np.zeros(C)

        # MATLAB: for j = 1:C (1-based indexing)
        # Convert to Python 0-based but maintain MATLAB logic
        for j in range(C):
            # MATLAB: isAnalysis = (drdp(j) <= 0) && (dpdt(j) < 0);
            isAnalysis = (drdp[j] <= 0) and (dpdt[j] < 0)
            if isAnalysis:
                count[j] = 1
                SlrWin[j] = SumSLR[j]
                PmaxminWin[j] = Pmaxmin[j]
                DownholeWinProp[j] = WinProp[j]
                
                # MATLAB: accumulate preceding non-analysis windows
                # k = j - 1; sS = 0; sP = 0;
                # while k >= 1
                k = j - 1  # Convert j to match MATLAB's j-1
                sS = 0  # sum of slurry
                sP = 0  # sum of pressure
                while k >= 0:  # MATLAB's k >= 1 becomes k >= 0 in Python
                    # MATLAB: if ~((drdp(k)<=0) && (dpdt(k)<0))
                    if not ((drdp[k] <= 0) and (dpdt[k] < 0)):
                        # MATLAB: sS = sS + PreWinSlurry(k); sP = sP + PreWinPmaxmin(k);
                        sS += PreWinSlurry[k]
                        sP += PreWinPmaxmin[k]
                    else:
                        break
                    k -= 1  # MATLAB: k = k - 1;
                
                SumPreWinSlurry[j] = sS
                SumPreWinPmaxmin[j] = sP
                
                # MATLAB: EnergyProxy(j) = sS * sP; EnergyDissipated(j) = SlrWin(j) * PmaxminWin(j);
                EnergyProxy[j] = sS * sP
                EnergyDissipated[j] = SlrWin[j] * PmaxminWin[j]

        # After calculating count array, get the number of windows
        num_windows = np.sum(count > 0)  # Count number of windows where count is 1

        # Create window iteration numbers for valid windows
        window_iterations = np.arange(1, num_windows + 1)

        # Save arrays data to CSV with window iteration including energy columns
        result_arrays_df = pd.DataFrame({
            'window_iteration': window_iterations,
            'SlrWin': SlrWin[count > 0],
            'PmaxminWin': PmaxminWin[count > 0],
            'DownholeWinProp': DownholeWinProp[count > 0],
            'EnergyProxy': EnergyProxy[count > 0],
            'EnergyDissipated': EnergyDissipated[count > 0]
        })

        output_array_file_path = os.path.join(user_dir, f'{stage}_arrays.csv')
        result_arrays_df.to_csv(output_array_file_path, index=False)

        # Statistics calculation (following MATLAB approach)
        Y = np.sum(count == 1)  # Number of analysis windows
        fracdecP = Y / C  # Fraction of decreasing pressure windows
        AnlWind = Y  # Analysis windows count
        
        # Original calculations
        TotalPres = np.nansum(PmaxminWin)
        TotalSlurryDP = np.nansum(SlrWin)
        TotalDHProp = np.nansum(DownholeWinProp)
        
        # Median calculations with proper masking (following MATLAB logic)
        medMaskDP = PmaxminWin > 5
        MedianDP = np.median(PmaxminWin[medMaskDP]) if np.any(medMaskDP) else np.nan
        MedianSlry = np.median(SlrWin[SlrWin > 0]) if np.any(SlrWin > 0) else np.nan
        
        FinalDH = DownholeWinProp[PmaxminWin > 0]
        MedDHProp = np.median(FinalDH[FinalDH > 0]) if np.any(FinalDH > 0) else np.nan
        
        # PPM calculations with safe division
        TotDHPPM = TotalDHProp / max(TotalPres, np.finfo(float).eps)
        DHPPM = MedDHProp / max(MedianDP, np.finfo(float).eps) if not np.isnan(MedianDP) else np.nan
        
        # Energy calculations (new MATLAB-derived attributes)
        NonZeroEnergyProxy = EnergyProxy[EnergyProxy > 0]
        NonZeroEnergyDissipated = EnergyDissipated[EnergyDissipated > 0]
        
        MedEnergyProxy = np.median(NonZeroEnergyProxy) if len(NonZeroEnergyProxy) > 0 else np.nan
        MedEnergyDissipated = np.median(NonZeroEnergyDissipated) if len(NonZeroEnergyDissipated) > 0 else np.nan
        MedRatio = MedEnergyDissipated / max(MedEnergyProxy, np.finfo(float).eps) if not np.isnan(MedEnergyProxy) else np.nan
        
        TotalEnergyProxy = np.nansum(EnergyProxy)
        TotalEnergyDissipated = np.nansum(EnergyDissipated)
        TotalEnergyRatio = TotalEnergyDissipated / max(TotalEnergyProxy, np.finfo(float).eps)

        # Compile all results including new MATLAB-derived attributes (excluding signal_denoised)
        file_results = [
            TotalPres, MedDHProp, MedianDP, DHPPM, TotDHPPM, TotalSlurryDP, MedianSlry, 
            stage, num_windows, TotalDHProp, MedEnergyProxy, MedEnergyDissipated, 
            MedRatio, TotalEnergyProxy, TotalEnergyDissipated, TotalEnergyRatio
        ]
        
        column_names = [
            'TEE', 'MedianDHPM', 'MedianDP', 'DownholePPM', 'TotalDHPPM', 'TotalSlurryDP', 
            'MedianSlurry', 'Stages', '# of Windows', 'TotalDHProp', 'MedEnergyProxy', 
            'MedEnergyDissipated', 'MedRatio', 'TotalEnergyProxy', 'TotalEnergyDissipated', 
            'TotalEnergyRatio'
        ]
        
        all_results = pd.concat([all_results, pd.DataFrame([file_results], columns=column_names)])

        # Clean up the temporary file after processing
        try:
            os.remove(temp_file_path)
        except Exception as e:
            st.warning(f"Could not remove temporary file {temp_file_path}: {e}")

    # Sort the consolidated results by the "Stages" column
    if not all_results.empty:
        all_results = all_results.sort_values(by='Stages', ascending=True)
    else:
        st.warning("No files were processed successfully.")

    # Save the consolidated results in session state
    st.session_state.data_prep['consolidated_output'] = all_results

    # Hide the progress bar
    progress_bar.empty()
    progress_text.empty()

def save_data_in_db(well_id, consolidated_output, user_id):
    try:
        # Check if inputs are valid
        if well_id is None:
            st.error("No well ID provided")
            return False
            
        if consolidated_output is None:
            st.error("No consolidated data available to save. Please process files first.")
            return False
            
        if not isinstance(consolidated_output, pd.DataFrame):
            st.error("Invalid data format. Expected a pandas DataFrame.")
            return False
            
        if consolidated_output.empty:
            st.error("Consolidated data is empty. No data to save.")
            return False

        # Convert DataFrame to records and update database
        modeling_data_summary = update_modeling_data(
            well_id, 
            consolidated_output.to_dict(orient='records'), 
            user_id
        )

        if modeling_data_summary is None:
            st.error("Failed to insert modeling data into the database.")
            return False

        # Display summary of inserted modeling data
        st.subheader("Database Operation Summary")
        st.write(f"Records updated or added: {len(modeling_data_summary)}")
        st.dataframe(pd.DataFrame(modeling_data_summary), use_container_width=True, hide_index=True)

        # Process and insert arrays data
        arrays_summary = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, item in enumerate(modeling_data_summary):
            stage = item['stage']
            data_id = item['data_id']

            arrays_file_path = os.path.join('export', str(user_id), f'{stage}_arrays.csv')
            if os.path.exists(arrays_file_path):
                try:
                    # Read and process arrays data
                    arrays_data = pd.read_csv(arrays_file_path)
                    last_non_zero_index = arrays_data.ne(0).any(axis=1)[::-1].idxmax()
                    arrays_data_filtered = arrays_data.loc[:last_non_zero_index].reset_index(drop=True)

                    # Update progress
                    progress_bar.progress((i + 1) / len(modeling_data_summary))
                    progress_text.text(f"Inserting arrays data for stage {stage}")

                    # Insert arrays data
                    stage_summary = call_insert_arrays_data(
                        data_id, 
                        arrays_data_filtered.to_dict(orient='records'),
                        user_id
                    )

                    if stage_summary['status'] == 'success':
                        arrays_summary.append({
                            'stage': stage, 
                            'records_added': stage_summary['records_added']
                        })
                    else:
                        st.error(f"Failed to insert arrays data for stage {stage}: {stage_summary['message']}")

                except Exception as e:
                    st.error(f"Error processing arrays data for stage {stage}: {str(e)}")
                    log_message(logging.ERROR, f"Error processing arrays data for stage {stage}: {str(e)}")

        # Display arrays data insertion summary
        st.subheader("Arrays Data Insertion Summary")
        if arrays_summary:
            summary_df = pd.DataFrame(arrays_summary)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No arrays data was inserted.")

        # Cleanup temporary files
        try:
            user_dir = os.path.join('export', str(user_id))
            if os.path.exists(user_dir):
                for file in os.listdir(user_dir):
                    os.remove(os.path.join(user_dir, file))
                os.rmdir(user_dir)
        except Exception as e:
            log_message(logging.WARNING, f"Error cleaning up temporary files: {str(e)}")

        st.success("Data successfully saved to database!")
        return True

    except Exception as e:
        st.error(f"Error saving data to database: {str(e)}")
        log_message(logging.ERROR, f"Error in save_data_in_db: {str(e)}")
        return False

    finally:
        # Clear progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'progress_text' in locals():
            progress_text.empty()

if __name__ == "__main__":
    main()