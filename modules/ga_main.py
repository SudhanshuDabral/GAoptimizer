import streamlit as st
import pandas as pd
import sys
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from utils.db import get_well_details, get_modeling_data

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ga.ga_calculation as ga_calculation
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

def main():
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.warning("Please login to access this page.")
        st.stop()

    initialize_state()
    st.title("Genetic Algorithm Optimizer for Regression Models (GA-ORM)")

    # Fetch well details from the database
    wells = get_well_details()

    # Allow the user to select a well
    well_options = {well['well_name']: well['well_id'] for well in wells}
    selected_well = st.selectbox("Select a Well", options=list(well_options.keys()))
    well_id = well_options[selected_well]

    # Fetch modeling data for the selected well
    data = get_modeling_data(well_id)

    # Display the fetched data in a table
    df = pd.DataFrame(data)
    df = df.sort_values(by='stage').drop(columns=['stage'])
    df['Productivity'] = ""

    st.write("Data Preview (You can edit the Productivity column):")

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("Productivity", editable=True)
    for col in df.columns:
        if col != "Productivity":
            gb.configure_column(col, editable=False)
    gb.configure_grid_options(domLayout='autoHeight', suppressMovableColumns=True, enableRangeSelection=True, clipboardDelimiter=',')

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

    # Z-Score Data button
    if st.button("Z-Score Data"):
        if edited_df['Productivity'].isnull().any() or (edited_df['Productivity'] == "").any():
            st.error("Please ensure all values in the 'Productivity' column are filled.")
        else:
            zscored_df = zscore_data(edited_df)
            st.session_state.zscored_df = zscored_df
            st.success("Data has been Z-Scored.")

    # Display Z-Scored Data if available
    if st.session_state.zscored_df is not None:
        st.write("Z-Scored Data Preview:")
        st.dataframe(st.session_state.zscored_df, use_container_width=True, hide_index=True)

    # Column selection
    drop_columns = st.multiselect("Select Columns to Drop", [col for col in df.columns if col != 'Productivity'])
    predictors = [col for col in df.columns if col != 'Productivity' and col not in drop_columns]

    # Model parameters
    r2_threshold = st.number_input("R² Threshold", min_value=0.0, max_value=1.0, value=0.55)
    coef_range = st.slider("Coefficient Range", -20.0, 20.0, (-10.0, 10.0))
    prob_crossover = st.number_input("Crossover Probability", min_value=0.0, max_value=1.0, value=0.8)
    prob_mutation = st.number_input("Mutation Probability", min_value=0.0, max_value=1.0, value=0.2)
    num_generations = st.number_input("Number of Generations", min_value=1, value=40)
    population_size = st.number_input("Population Size", min_value=1, value=50)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start GA Optimization", key="start_button", disabled=st.session_state.running):
            with st.spinner('Running Genetic Algorithm...'):
                st.session_state.running = True
                st.session_state.edited_df = edited_df
                st.rerun()

    with col2:
        if st.button("Stop GA Optimization", key="stop_button", disabled=not st.session_state.running):
            st.session_state.running = False
            st.rerun()

    if st.session_state.running:
        start_ga_optimization(st.session_state.zscored_df, 'Productivity', predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size)

    # Display results if available
    if st.session_state.result:
        best_ind, best_r2_score, response_equation, selected_feature_names, errors_df = st.session_state.result
        st.success("Genetic Algorithm Optimization Complete!")
        st.write("Best individual with R² score of:", best_r2_score)
        st.write("Response Equation:", response_equation)
        st.write("Selected Features:", selected_feature_names)
        st.write("Error Table for Individual Data Points:")
        st.write(errors_df)

def start_ga_optimization(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size):
    # Convert all columns to numeric to avoid type issues
    df = df.apply(pd.to_numeric, errors='coerce')

    start_time = time.time()
    timer_placeholder = st.empty()

    result = ga_calculation.run_ga(
        df, target_column, predictors, r2_threshold, coef_range,
        prob_crossover, prob_mutation, num_generations, population_size,
        timer_placeholder
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
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric and coerce errors to NaN
        col_mean = df[column].mean()
        col_std = df[column].std(ddof=1)  # Sample standard deviation
        df[column] = (df[column] - col_mean) / col_std
    return df

if __name__ == "__main__":
    main()
