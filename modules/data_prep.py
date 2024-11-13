import json
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from utils.db import (update_modeling_data, call_insert_arrays_data, 
                      create_new_well, get_well_details, bulk_insert_well_completion_records, log_message)

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
        data = pd.read_csv(temp_file_path)
        # Ensure timestamp_utc is in datetime format and calculate seconds from the first timestamp
        data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'])

        data['seconds'] = (data['timestamp_utc'] - data['timestamp_utc'].min()).dt.total_seconds()
    
        # Extract data using column headers
        T = data['seconds'].values
        SlRate = data['slurry_rate'].values
        Pres = data['treating_pressure'].values
        DownHoleProp = data['bottomhole_prop_mass'].values

        completion_data = pd.DataFrame({
        'treating_pressure': data['treating_pressure'],
        'slurry_rate': data['slurry_rate'],
        'bottomhole_prop_mass': data['bottomhole_prop_mass'],
        'time_seconds': data['seconds'].astype(int),
        'epoch': data['epoch']  # Use the original timestamp_utc value
    })

        insert_result = bulk_insert_well_completion_records(st.session_state.data_prep['well_id'], stage, completion_data, user_id)

        if insert_result['status'] == 'error':
            st.error(f"Failed to insert well completion records for stage {stage}: {insert_result['message']}")
            continue
        

        # Filter by SlRate > 2
        IndexS = np.where(SlRate > 2)[0]
        T = T[IndexS]
        SlRate = SlRate[IndexS]
        Pres = Pres[IndexS]
        DownHoleProp = DownHoleProp[IndexS]

        n = len(T)

        WS = 11  # Window size, should always be odd
        MP = (WS + 1) // 2

        # Initialize arrays
        LI = np.zeros(n, dtype=int)
        UL = np.zeros(n, dtype=int)
        SlpeSR = np.zeros((2, n))  # polyfit results have 2 coefficients
        SumSLR = np.zeros(n)
        WinProp = np.zeros(n)
        Pmaxmin = np.zeros(n)
        SlpeP = np.zeros((2, n))

        for i in range(n):
            LI[i] = (MP + (i - 1) * (MP - 1))-1
            UL[i] = (MP + (i - 1) * (MP - 1) + WS - 1)
            if UL[i] > n:
                break
            coef_SR = np.polyfit(T[LI[i]:UL[i]], SlRate[LI[i]:UL[i]], 1)
            coef_P = np.polyfit(T[LI[i]:UL[i]], Pres[LI[i]:UL[i]], 1)
            SlpeSR[:, i] = coef_SR
            SlpeP[:, i] = coef_P
            SumSLR[i] = np.sum(SlRate[LI[i]:UL[i]]) * 0.7
            WinProp[i] = np.sum(DownHoleProp[LI[i]:UL[i]])
            Pmaxmin[i] = np.max(Pres[LI[i]:UL[i]]) - np.min(Pres[LI[i]:UL[i]])

        # Additional processing and result storage
        SumSLR = SumSLR.T
        Pmaxmin = Pmaxmin.T
        dpdt = SlpeP[0, :]
        drdp = SlpeSR[0, :] / SlpeP[0, :]
        dpdt = dpdt.T
        drdp = drdp.T

        C = len(drdp)
        count = np.zeros(C)
        PropsWin = np.zeros(C)
        SlrWin = np.zeros(C)
        PmaxminWin = np.zeros(C)
        DownholeWinProp = np.zeros(C)

        for j in range(C):
            if drdp[j] <= 0 and dpdt[j] < 0:
                count[j] = 1
                SlrWin[j] = SumSLR[j]
                PmaxminWin[j] = Pmaxmin[j]
                DownholeWinProp[j] = WinProp[j]

        # Save arrays data to CSV
        result_arrays_df = pd.DataFrame({
            'SlrWin': SlrWin,
            'PmaxminWin': PmaxminWin,
            'DownholeWinProp': DownholeWinProp
        })

        output_array_file_path = os.path.join(user_dir, f'{stage}_arrays.csv')
        result_arrays_df.to_csv(output_array_file_path, index=False)

        TotalPres = np.sum(PmaxminWin)
        TotalSlurryDP = np.sum(SlrWin)
        TotalDHProp = np.sum(DownholeWinProp)
        MedianDP = np.median(PmaxminWin[PmaxminWin > 5])
        MedianSlry = np.median(SlrWin[SlrWin > 0])
        FinalDownHoleWinProp = DownholeWinProp[PmaxminWin > 0]
        MedDHProp = np.median(FinalDownHoleWinProp[FinalDownHoleWinProp > 0])

        TotDHPPM = TotalDHProp / TotalPres
        DHPPM = MedDHProp / MedianDP

        file_results = [TotalPres, MedDHProp, MedianDP, DHPPM, TotDHPPM, TotalSlurryDP, MedianSlry, stage]
        all_results = pd.concat([all_results, pd.DataFrame([file_results], columns=['TEE', 'MedianDHPM', 'MedianDP', 'DownholePPM', 'TotalDHPPM', 'TotalSlurryDP', 'MedianSlurry', 'Stages'])])

    # Sort the consolidated results by the "Stages" column
    all_results = all_results.sort_values(by='Stages', ascending=True)

    # Writing the consolidated results to an Excel file
    output_file_path = os.path.join(user_dir, 'consolidated_output.csv')
    all_results.to_csv(output_file_path, index=False)

    # Save the consolidated results in session state
    st.session_state.data_prep['consolidated_output'] = all_results

    # Hide the progress bar
    progress_bar.empty()
    progress_text.empty()

    st.success(f"Consolidated data exported successfully to {output_file_path}")
    st.download_button(
        label="Download consolidated output",
        data=open(output_file_path, "rb").read(),
        file_name="consolidated_output.csv",
        mime="text/csv"
    )

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