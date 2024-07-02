import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
import pandas as pd

r2_values = []
iterations = []

def update_plot(iterations, r2_values, plot_placeholder):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=r2_values,
                             mode='lines+markers',
                             name='R² vs Iteration'))

    fig.update_layout(title='R² Values vs Iteration',
                      xaxis_title='Iteration',
                      yaxis_title='R² Value')

    plot_placeholder.plotly_chart(fig, use_container_width=True)

def plot_r2_vs_iteration():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state['iterations'], y=st.session_state['r2_values'],
                             mode='lines+markers',
                             name='R² vs Iteration'))

    fig.update_layout(title='R² Values vs Iteration',
                      xaxis_title='Iteration',
                      yaxis_title='R² Value')

    st.plotly_chart(fig, use_container_width=True)

def plot_column(df, column_name, stage):
    """
    Create a line plot for a specific column in the DataFrame.
    
    :param df: pandas DataFrame containing the data
    :param column_name: str, name of the column to plot
    :return: plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column_name],
        mode='lines+markers',
        name=column_name
    ))
    
    fig.update_layout(
        title=f'{column_name} for Stage {stage}',
        xaxis_title='Index',
        yaxis_title=column_name,
        height=500,  # Fixed height for consistency
    )
    
    return fig

#function to plot the actual vs predicted productivity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_actual_vs_predicted(error_df):
    fig = go.Figure()

    # Get unique well names and assign colors
    unique_wells = error_df['WellName'].unique()
    colors = px.colors.qualitative.Plotly[:len(unique_wells)]
    color_map = dict(zip(unique_wells, colors))

    # Plot data points for each well
    for well in unique_wells:
        well_data = error_df[error_df['WellName'] == well]
        fig.add_trace(go.Scatter(
            x=well_data['Predicted'],
            y=well_data['Actual'],
            mode='markers',
            name=well,
            text=[f"{well}_Stage{stage}" for stage in well_data['stage']],  # Add well name and stage to hover text
            hoverinfo='text+x+y',
            marker=dict(color=color_map[well], size=8)
        ))

    # Calculate regression line
    X = error_df['Predicted'].values.reshape(-1, 1)
    y = error_df['Actual'].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Calculate R² value
    r2 = r2_score(y, y_pred)

    # Add regression line
    fig.add_trace(go.Scatter(
        x=error_df['Predicted'],
        y=y_pred,
        mode='lines',
        name=f'Regression Line (R² = {r2:.4f})',
        line=dict(color='red', dash='dash')
    ))

    # Add perfect prediction line
    max_value = max(error_df['Predicted'].max(), error_df['Actual'].max())
    min_value = min(error_df['Predicted'].min(), error_df['Actual'].min())
    # fig.add_trace(go.Scatter(
    #     x=[min_value, max_value],
    #     y=[min_value, max_value],
    #     mode='lines',
    #     name='Perfect Prediction',
    #     line=dict(color='white', dash='dot')
    # ))

    fig.update_layout(
        title='Model Performance Plot',
        xaxis_title='Predicted Productivity',
        yaxis_title='Actual Productivity',
        legend_title='Wells',
        height=600,
        width=800
    )

    # Make the plot square and set axis ranges to be equal
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(range=[min_value, max_value]),
        yaxis_range=[min_value, max_value]
    )

    return fig