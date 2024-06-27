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
def plot_actual_vs_predicted(actual, predicted, stages, excluded_rows):
    fig = go.Figure()

    # Create a mask for excluded points
    excluded_mask = stages.index.isin(excluded_rows)
    
    # Plot predicted productivity (all points)
    fig.add_trace(go.Scatter(
        x=stages,
        y=predicted,
        mode='lines+markers',
        name='Predicted Productivity',
        line=dict(color='green')
    ))

    # Plot actual productivity (included points)
    fig.add_trace(go.Scatter(
        x=stages[~excluded_mask],
        y=actual[~excluded_mask],
        mode='lines+markers',
        name='Actual Productivity (Included)',
        line=dict(color='blue')
    ))
    
    # Plot actual productivity (excluded points)
    fig.add_trace(go.Scatter(
        x=stages[excluded_mask],
        y=actual[excluded_mask],
        mode='markers',
        name='Actual Productivity (Excluded)',
        marker=dict(color='red', size=10, symbol='x')
    ))

    # Add labels for excluded points
    for i, stage in enumerate(stages[excluded_mask]):
        fig.add_annotation(
            x=stage,
            y=actual[excluded_mask].iloc[i],
            text=f"Stage {stage}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=20,
            ay=-30
        )

    fig.update_layout(
        title='Model Fitness Plot',
        xaxis_title='Stage',
        yaxis_title='Productivity',
        legend_title='Legend',
        height=600,
        width=800
    )
    
    return fig