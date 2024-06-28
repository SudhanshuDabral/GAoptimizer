import plotly.graph_objects as go
import streamlit as st
import plotly.express as px

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
def plot_actual_vs_predicted(actual, predicted, excluded_rows, well_names, stages):
    fig = go.Figure()

    # Create a mask for excluded points
    excluded_mask = stages.index.isin(excluded_rows)
    
    # Get unique well names
    unique_wells = well_names.unique()
    colors = px.colors.qualitative.Plotly[:len(unique_wells)]
    color_map = dict(zip(unique_wells, colors))

    # Plot included points for each well
    for well in unique_wells:
        well_mask = (well_names == well) & (~excluded_mask)
        fig.add_trace(go.Scatter(
            x=predicted[well_mask],
            y=actual[well_mask],
            mode='markers',
            name=f'{well} (Included)',
            marker=dict(color=color_map[well], size=8)
        ))
    
    # Plot excluded points for each well
    for well in unique_wells:
        well_mask = (well_names == well) & excluded_mask
        fig.add_trace(go.Scatter(
            x=predicted[well_mask],
            y=actual[well_mask],
            mode='markers',
            name=f'{well} (Excluded)',
            marker=dict(color=color_map[well], size=10, symbol='x')
        ))

    # Add perfect prediction line
    max_value = max(actual.max(), predicted.max())
    min_value = min(actual.min(), predicted.min())
    fig.add_trace(go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash')
    ))

    # Add labels for excluded points
    for idx in stages.index[excluded_mask]:
        fig.add_annotation(
            x=predicted[idx],
            y=actual[idx],
            text=f"{well_names[idx]}: Stage {stages[idx]}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color_map[well_names[idx]],
            ax=20,
            ay=-30
        )

    fig.update_layout(
        title='Model Fitness',
        xaxis_title='Predicted Productivity',
        yaxis_title='Actual Productivity',
        legend_title='Legend',
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