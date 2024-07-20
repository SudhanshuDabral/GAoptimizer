import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.db import fetch_all_models_with_users, fetch_model_details, fetch_training_data, fetch_sensitivity_data
from utils.plotting import plot_actual_vs_predicted, create_tornado_chart, create_feature_importance_chart, create_elasticity_analysis
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime, timedelta

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
    logger.log(level, f"[Model_Explorer] {message}")

def create_metric_card(title, value, delta=None):
    card_id = f"card_{title.lower().replace(' ', '_')}"
    st.markdown(
        f"""
        <style>
        @keyframes flipIn {{
            from {{
                transform: perspective(400px) rotateY(90deg);
                opacity: 0;
            }}
            to {{
                transform: perspective(400px) rotateY(0deg);
                opacity: 1;
            }}
        }}
        #{card_id} {{
            animation: flipIn 0.5s ease-out;
        }}
        </style>
        <div id="{card_id}" style="
            background-color: #2D2D2D;
            border: 1px solid #FF6D00;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out;
        "
        onmouseover="this.style.transform='scale(1.05)'"
        onmouseout="this.style.transform='scale(1)'"
        >
            <h3 style="margin-top: 0; color: #FF6D00;">{title}</h3>
            <p style="font-size: 24px; font-weight: bold; margin-bottom: 5px; color: #F5F5F5;">{value}</p>
            {f'<p style="color: {"#4CAF50" if float(delta) > 0 else "#FF5252"};">{delta}</p>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_tag(text, color="#FF6D00"):
    return f'<span style="background-color: {color}; color: #FFFFFF; padding: 5px 10px; margin: 3px; border-radius: 10px; display: inline-block; font-size: 16px;">{text}</span>'

def filter_models(models, name_filter, user_filter, start_date, end_date):
    filtered_models = models
    if name_filter:
        filtered_models = [m for m in filtered_models if name_filter.lower() in m['model_name'].lower()]
    if user_filter:
        filtered_models = [m for m in filtered_models if user_filter.lower() in m['created_by'].lower()]
    if start_date:
        filtered_models = [m for m in filtered_models if m['created_on'].date() >= start_date]
    if end_date:
        filtered_models = [m for m in filtered_models if m['created_on'].date() <= end_date]
    return filtered_models

def main(authentication_status):
    log_message(logging.INFO, "Entering Model Explorer main function")
    if not authentication_status:
        st.warning("Please login to access this page.")
        log_message(logging.WARNING, "Unauthenticated access attempt to Model Explorer")
        return

    st.title("Model Explorer")

    # Fetch all saved models with user information
    log_message(logging.INFO, "Fetching all saved models with user information")
    saved_models = fetch_all_models_with_users()

    if not saved_models:
        st.warning("No saved models found. Please create and save a model in the GA Optimizer page.")
        log_message(logging.WARNING, "No saved models found")
        return

    # Model selection with search
    st.subheader("Model Selection")
    
    # Create a unique key for the session state
    if 'model_search' not in st.session_state:
        st.session_state.model_search = ''
    if 'show_advanced_filters' not in st.session_state:
        st.session_state.show_advanced_filters = False
    if 'filtered_models' not in st.session_state:
        st.session_state.filtered_models = saved_models

    # Search input and advanced filter button
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("Search models", key="model_search")
    with col2:
        if st.button("Advanced Filters"):
            st.session_state.show_advanced_filters = not st.session_state.show_advanced_filters

    # Advanced filters panel
    if st.session_state.show_advanced_filters:
        with st.sidebar:
            st.subheader("Advanced Filters")
            user_filter = st.text_input("Filter by User")
            start_date = st.date_input("Start Date", value=None)
            end_date = st.date_input("End Date", value=None)
            if st.button("Apply Filters"):
                st.session_state.filtered_models = filter_models(saved_models, search_term, user_filter, start_date, end_date)

    # Filter models based on search term
    if search_term:
        st.session_state.filtered_models = [m for m in st.session_state.filtered_models if search_term.lower() in m['model_name'].lower()]

    # Model selection dropdown
    model_options = [f"{model['model_name']} (by {model['created_by']} on {model['created_on']})" for model in st.session_state.filtered_models]
    selected_model = st.selectbox("Select a model", options=model_options)

    if selected_model:
        model_name = selected_model.split(" (by ")[0]
        log_message(logging.INFO, f"User selected model: {model_name}")
        model_id = next(model['model_id'] for model in st.session_state.filtered_models if model['model_name'] == model_name)
        
        # Fetch model details
        log_message(logging.INFO, f"Fetching details for model: {model_id}")
        model_details = fetch_model_details(model_id)
        
        if model_details:
            try:
                st.header("Model Summary Dashboard")

                # Model Overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    create_metric_card("Model Name", model_details['model_name'])
                with col2:
                    create_metric_card("Regression Type", model_details['regression_type'])
                with col3:
                    create_metric_card("Created On", model_details['created_on'].strftime("%Y-%m-%d %H:%M:%S"))

                # Model Performance
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    create_metric_card("Weighted R² Score", f"{model_details['weighted_r2_score']:.4f}")
                with col2:
                    create_metric_card("Full Dataset R² Score", f"{model_details['full_dataset_r2_score']:.4f}")

                # Model Equation
                st.subheader("Model Equation")
                st.code(model_details['equation'])

                # Tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(["Model Parameters", "Features", "Training Data", "Sensitivity Analysis"])

                with tab1:
                    st.subheader("Model Parameters")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        create_metric_card("R² Threshold", f"{model_details['r2_threshold']:.2f}")
                    with col2:
                        create_metric_card("Crossover Probability", f"{model_details['probability_crossover']:.2f}")
                    with col3:
                        create_metric_card("Mutation Probability", f"{model_details['probability_mutation']:.2f}")
                    with col4:
                        create_metric_card("Population Size", str(model_details['population_size']))

                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Predictors")
                        predictor_tags = "".join([create_tag(pred) for pred in model_details['predictors']])
                        st.markdown(predictor_tags, unsafe_allow_html=True)
                    with col2:
                        st.subheader("Selected Features")
                        feature_tags = "".join([create_tag(feat) for feat in model_details['selected_feature_names']])
                        st.markdown(feature_tags, unsafe_allow_html=True)

                with tab3:
                    st.subheader("Training Data")
                    log_message(logging.INFO, f"Fetching training data for model: {model_id}")
                    training_data = fetch_training_data(model_id)
                    if training_data is not None and not training_data.empty:
                        # Remove unnecessary columns
                        columns_to_remove = [col for col in training_data.columns if col.endswith('_id') or col in ['created_on']]
                        training_data_display = training_data.drop(columns=columns_to_remove)

                        # Highlight excluded rows
                        def highlight_excluded(row):
                            return ['background-color: #FF6D00' if row['is_excluded'] else '' for _ in row]

                        # Display the training data with highlighting
                        st.dataframe(training_data_display.style.apply(highlight_excluded, axis=1), hide_index=True)
                        
                        st.markdown("**Note:** Highlighted rows/stages were excluded from modeling.")
                    
                        # Actual vs Predicted Plot
                        st.subheader("Actual vs Predicted Plot")
                        fig = plot_actual_vs_predicted(training_data_display)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No training data available for this model.")

                with tab4:
                    st.subheader("Sensitivity Analysis")
                    log_message(logging.INFO, f"Fetching sensitivity data for model: {model_id}")
                    sensitivity_data = fetch_sensitivity_data(model_id)
                    if not sensitivity_data.empty:
                        # Create and display sensitivity charts
                        baseline_productivity = sensitivity_data['min_productivity'].mean()  # This is an approximation
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_tornado = create_tornado_chart(sensitivity_data, baseline_productivity)
                            st.plotly_chart(fig_tornado, use_container_width=True)
                        
                        with col2:
                            fig_importance = create_feature_importance_chart(sensitivity_data)
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                        fig_elasticity = create_elasticity_analysis(sensitivity_data, baseline_productivity)
                        st.plotly_chart(fig_elasticity, use_container_width=True)
                    else:
                        st.warning("No sensitivity data available for this model.")

            except KeyError as e:
                st.error(f"Error: Missing expected data in model details. KeyError: {str(e)}")
                log_message(logging.ERROR, f"KeyError when processing model details for model_id {model_id}: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred while processing model details: {str(e)}")
                log_message(logging.ERROR, f"Unexpected error when processing model details for model_id {model_id}: {str(e)}")
        else:
            st.error("Failed to fetch model details. Please try again or contact support.")
            log_message(logging.ERROR, f"Failed to fetch details for model: {model_id}")

    log_message(logging.INFO, "Exiting Model Explorer main function")

if __name__ == "__main__":
    main(st.session_state.get("authentication_status"))