import json
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from utils.db import call_insert_or_update_well_and_consolidated_output, call_insert_arrays_data

def main():
    # Check if user is authenticated
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.warning("Please login to access this page.")
        st.stop()

    st.title("Data Preparation")
    st.write("Provide the well details and upload CSV files for each stage of the well completion.")

    # Form to input well details
    if 'well_details' not in st.session_state:
        with st.form("well_details_form"):
            well_name = st.text_input("Well Name")
            well_api = st.text_input("Well API")
            latitude = st.text_input("Latitude")
            longitude = st.text_input("Longitude")
            tvd = st.text_input("TVD (ft)")
            reservoir_pressure = st.text_input("Reservoir Pressure")

            # Submit button
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

                # If all validations pass, save the well details
                if (
                    len(well_name) <= 255 and well_api.isdigit() and
                    (-90 <= latitude <= 90) and (-180 <= longitude <= 180) and
                    (500 <= tvd <= 15500) and (2000 <= reservoir_pressure <= 8500)
                ):
                    st.session_state.well_details = {
                        "Well Name": well_name,
                        "Well API": well_api,
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "TVD (ft)": tvd,
                        "Reservoir Pressure": reservoir_pressure
                    }
                    st.experimental_rerun()

    if 'well_details' in st.session_state:
        well_details = st.session_state.well_details
        st.subheader("Well Information")
        st.table(pd.DataFrame([well_details]))

        st.write("Well details submitted. Now upload the CSV files for each stage.")
        uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

        if uploaded_files:
            if 'processing_files' not in st.session_state:
                st.session_state.processing_files = True
                process_files(uploaded_files, well_details)
                st.session_state.processing_files = False

            # Show the consolidated results in a table
            if 'consolidated_output' in st.session_state:
                st.subheader("Data for Modelling")
                st.dataframe(st.session_state['consolidated_output'], use_container_width=True)

            # Add a button to save data in DB
            if st.button("Save Data in DB"):
                save_data_in_db(well_details, st.session_state['consolidated_output'], st.session_state['user_id'])

def extract_stage(filename):
    match = re.search(r'_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    else:
        return filename

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

        Y = np.sum(count == 1)
        fracdecP = Y / C
        SumCTemp = np.sum(count)

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
    st.session_state['consolidated_output'] = all_results

    # Show the consolidated results in a table
    st.subheader("Data for Modeling")
    st.dataframe(all_results, use_container_width=True)

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

def save_data_in_db(well_details, consolidated_output, user_id):
    # Insert well information and consolidated output
    summary = call_insert_or_update_well_and_consolidated_output(
        well_details, consolidated_output.to_dict(orient='records'), user_id)

    if summary is None:
        st.error("Failed to insert well information and consolidated output into the database.")
        return

    # Consolidated summary for well information and data for modeling
    st.subheader("Database Operation Summary")
    st.write(f"Records updated or added: {len(consolidated_output)}")

    # Extract data_ids for each stage
    data_ids = summary  # Assuming the summary is a list of dictionaries
    arrays_summary = []

    # Initialize the progress bar for the insertion process
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Process and insert arrays data for each stage
    for i, item in enumerate(data_ids):
        stage = item['stage']
        data_id = item['data_id']

        arrays_file_path = os.path.join('export', str(user_id), f'{stage}_arrays.csv')
        if os.path.exists(arrays_file_path):
            arrays_data = pd.read_csv(arrays_file_path).to_dict(orient='records')

            # Update the progress bar
            progress_bar.progress((i + 1) / len(data_ids))
            progress_text.text(f"Inserting arrays data for stage {stage}")

            # Call the function to insert arrays data
            stage_summary = call_insert_arrays_data(data_id, arrays_data, user_id)
            arrays_summary.append({'stage': stage, 'records_added': stage_summary['records_added']})

    # Display consolidated arrays data insertion summary
    st.subheader("Arrays Data Insertion Summary")
    for summary in arrays_summary:
        st.write(f"Stage {summary['stage']}: {summary['records_added']} records added")

    # Cleanup: Remove the user-specific directory
    user_dir = os.path.join('export', str(user_id))
    if os.path.exists(user_dir):
        for file in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, file))
        os.rmdir(user_dir)

if __name__ == "__main__":
    main()

