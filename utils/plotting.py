import plotly.graph_objects as go
import streamlit as st

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

def plot_column(df, column_name):
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
        title=f'{column_name} vs Index',
        xaxis_title='Index',
        yaxis_title=column_name,
        height=600,
        width=800
    )
    
    return fig