import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ga.ga_calculation as ga_calculation
import time


def initialize_state():
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'result' not in st.session_state:
        st.session_state.result = None

def main():
    initialize_state()
    st.title("Genetic Algorithm Optimizer for Regression Models (GA-ORM)")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Column selection
        target_column = st.selectbox("Select Target Column", df.columns)
        drop_columns = st.multiselect("Select Columns to Drop", df.columns)

        predictors = [col for col in df.columns if col != target_column and col not in drop_columns]

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
                    st.experimental_rerun()

        with col2:
            if st.button("Stop GA Optimization", key="stop_button", disabled=not st.session_state.running):
                st.session_state.running = False
                st.experimental_rerun()

        if st.session_state.running:
            start_ga_optimization(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size)

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
    st.experimental_rerun()

if __name__ == "__main__":
    main()
