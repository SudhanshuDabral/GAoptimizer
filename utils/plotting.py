import traceback
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
from logging.handlers import RotatingFileHandler
import logging
import os
import numpy as np
import pandas as pd
from itertools import cycle


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
    logger.log(level, f"[plotting] {message}")

r2_values = []
iterations = []

def update_plot(iterations, r2_values, plot_placeholder, model_markers):
    fig = go.Figure()

    # Add the main trace for R² values
    fig.add_trace(go.Scatter(
        x=iterations,
        y=r2_values,
        mode='lines+markers',
        name='R² vs Iteration',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))

    # Add markers for model creation points
    for model_num, (iteration, r2) in model_markers.items():
        fig.add_trace(go.Scatter(
            x=[iteration],
            y=[r2],
            mode='markers',
            name=f'Model {model_num}',
            marker=dict(size=12, symbol='star', color='red')
        ))

    fig.update_layout(
        title='R² Values vs Iteration (All Models)',
        xaxis_title='Iteration',
        yaxis_title='R² Value',
        showlegend=True
    )

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
    
def plot_column(df, stage):
    """
    Create a line plot for Productivity, total_dhppm_stage, and total_slurry_dp_stage.
    
    :param df: pandas DataFrame containing the data
    :param stage: str or int, the stage number
    :return: plotly Figure object
    """
    fig = go.Figure()

    columns_to_plot = ['Productivity', 'total_dhppm_stage', 'total_slurry_dp_stage']
    colors = ['green', 'red', 'blue']

    for i, (column, color) in enumerate(zip(columns_to_plot, colors)):
        if column in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[column],
                mode='lines+markers',
                name=column,
                line=dict(color=color),
                yaxis='y' if i == 0 else 'y2'
            ))
        else:
            print(f"Warning: Column '{column}' not found in the DataFrame")

    fig.update_layout(
        title=f'Productivity, Total DHPPM, and Total Slurry DP for Stage {stage}',
        xaxis_title='Index',
        yaxis=dict(
            title='Productivity',
            titlefont=dict(color="green"),
            tickfont=dict(color="green")
        ),
        yaxis2=dict(
            title='DHPPM / Slurry DP',
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying='y',
            side='right'
        ),
        height=500,  # Fixed height for consistency
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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

#function to create a tornado chart for model sensitivity
def create_tornado_chart(sensitivity_df, baseline_productivity):
    sensitivity_df['Min_Diff'] = baseline_productivity - sensitivity_df['Min Productivity']
    sensitivity_df['Max_Diff'] = sensitivity_df['Max Productivity'] - baseline_productivity
    
    sensitivity_df = sensitivity_df.sort_values('Max_Diff', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sensitivity_df['Attribute'],
        x=sensitivity_df['Min_Diff'],
        name='Minimum',
        orientation='h',
        marker=dict(color='blue')
    ))
    
    fig.add_trace(go.Bar(
        y=sensitivity_df['Attribute'],
        x=sensitivity_df['Max_Diff'],
        name='Maximum',
        orientation='h',
        marker=dict(color='red')
    ))
    
    fig.update_layout(
        title='Tornado Chart - Model Sensitivity',
        xaxis_title='Change in Productivity',
        yaxis_title='Attributes',
        barmode='relative',
        height=500,
        width=800
    )
    
    return fig

def create_feature_importance_chart(sensitivity_df):
    sensitivity_df['Impact_Range'] = sensitivity_df['Max Productivity'] - sensitivity_df['Min Productivity']
    sensitivity_df = sensitivity_df.sort_values('Impact_Range', ascending=True)
    
    fig = go.Figure(go.Bar(
        y=sensitivity_df['Attribute'],
        x=sensitivity_df['Impact_Range'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Impact Range on Productivity',
        yaxis_title='Attributes',
        height=400
    )
    
    return fig

def create_elasticity_analysis(sensitivity_df, zscored_statistics, baseline_productivity):
    elasticity = []
    for _, row in sensitivity_df.iterrows():
        attr = row['Attribute']
        median_value = zscored_statistics[attr]['median']
        max_value = row['Max Productivity']
        min_value = row['Min Productivity']
        
        pct_change_attr = (zscored_statistics[attr]['max'] - zscored_statistics[attr]['min']) / median_value
        pct_change_prod = (max_value - min_value) / baseline_productivity
        
        elasticity.append(pct_change_prod / pct_change_attr if pct_change_attr != 0 else 0)

    sensitivity_df['Elasticity'] = elasticity
    sensitivity_df = sensitivity_df.sort_values('Elasticity', ascending=True)

    fig = go.Figure(go.Bar(
        y=sensitivity_df['Attribute'],
        x=sensitivity_df['Elasticity'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Elasticity Analysis',
        xaxis_title='Elasticity',
        yaxis_title='Attributes',
        height=400
    )
    
    return fig


def plot_sensitivity_results(sensitivity_results, attribute_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sensitivity_results['TestPoint'],
        y=sensitivity_results['Productivity'],
        mode='lines+markers',
        name='Productivity'
    ))

    fig.update_layout(
        title=f'Sensitivity Analysis for {attribute_name}',
        xaxis_title=f'{attribute_name} (Z-score)',
        yaxis_title='Productivity',
        height=500,
        width=800
    )

    return fig

def create_influence_chart(influence_df):
    """
    Create a bar chart showing the influence of attributes on productivity.
    
    :param influence_df: DataFrame containing 'Attribute' and 'Influence' columns
    :return: Plotly Figure object
    """
    fig = px.bar(influence_df, 
                 x='Attribute', 
                 y='Influence', 
                 title='Attribute Influence on Productivity',
                 labels={'Influence': 'Productivity Range'},
                 color='Influence',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_title="Attribute",
        yaxis_title="Productivity Range",
        coloraxis_colorbar_title="Influence"
    )
    
    return fig


def create_multi_axis_plot(df, title, event_windows, leakoff_periods):
    try:
        log_message(logging.INFO, f"Starting to create plot for {title}")
        
        # Create figure with multiple y-axes
        fig = make_subplots(rows=1, cols=1)
        log_message(logging.INFO, "Created subplot")

        # Define colors for each trace
        colors = {
            'Treating Pressure': 'rgb(255, 99, 71)',
            'Slurry Rate': 'rgb(255, 70, 51)',
            'BH Prop Mass': 'rgb(162, 28, 141)',
            'PPC': 'rgb(60, 179, 113)'
        }

        # Set a fixed number of ticks for all axes
        num_ticks = 6

        # Add traces
        for i, (column, color) in enumerate(colors.items()):
            fig.add_trace(
                go.Scatter(
                    x=df['Time Seconds'], 
                    y=df[column], 
                    name=column, 
                    line=dict(color=color),
                    yaxis=f'y{i+1}'
                )
            )
            log_message(logging.INFO, f"Added trace for {column}")

        # Highlight event windows
        log_message(logging.INFO, f"Starting to add {len(event_windows)} event windows")
        for i, (start, end) in enumerate(event_windows):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
            )
            if i % 100 == 0:
                log_message(logging.INFO, f"Added {i+1}/{len(event_windows)} event windows")
        log_message(logging.INFO, "Finished adding event windows")

        # Add horizontal bars for leakoff periods
        log_message(logging.INFO, f"Starting to add {len(leakoff_periods)} leakoff periods")
        for i, (start, end) in enumerate(leakoff_periods):
            fig.add_shape(
                type="rect",
                x0=start, x1=end,
                y0=0, y1=1,
                yref="paper",
                fillcolor="lightblue", opacity=0.3,
                layer="below", line_width=0,
            )
            if i % 100 == 0:
                log_message(logging.INFO, f"Added {i+1}/{len(leakoff_periods)} leakoff periods")
        log_message(logging.INFO, "Finished adding leakoff periods")

        # Ensure 'Time Seconds' is numeric
        df['Time Seconds'] = pd.to_numeric(df['Time Seconds'], errors='coerce')

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=50, r=150, t=50, b=50),
            xaxis=dict(
                domain=[0, 0.8],
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.1)',
                griddash='dash',
                title_text="Time (seconds)"
            ),
            height=600,
        )
        log_message(logging.INFO, "Updated layout")

        # Update y-axes
        for i, (column, color) in enumerate(colors.items()):
            min_val = df[column].min()
            max_val = df[column].max()
            
            fig.update_layout(**{
                f'yaxis{i+1 if i > 0 else ""}': dict(
                    title_text=column,
                    titlefont=dict(color=color),
                    tickfont=dict(color=color),
                    tickmode='array',
                    tickvals=np.linspace(min_val, max_val, num_ticks),
                    tickformat='.6f',
                    anchor="free",
                    overlaying="y" if i > 0 else None,
                    side="right",
                    position=1 - (i * 0.05),
                    showgrid=True,
                    gridcolor='rgba(211, 211, 211, 0.1)',
                    griddash='dash',
                    zeroline=False,
                    showline=True,
                    linecolor=color,
                    linewidth=2,
                    range=[min_val, max_val]
                )
            })
            log_message(logging.INFO, f"Updated y-axis for {column}")

        log_message(logging.INFO, f"Successfully created multi-axis plot for {title}")
        return fig
    except Exception as e:
        log_message(logging.ERROR, f"Error creating multi-axis plot: {str(e)}")
        raise

import colorsys

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.8, 0.9) for x in range(n)]
    return list(map(lambda x: f'rgb{tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*x))}', HSV_tuples))

def plot_rolling_ir(combined_data, well_name):
    try:
        fig = go.Figure()
        
        unique_stages = combined_data['stage'].unique()
        num_stages = len(unique_stages)
        logging.info(f"Plotting {num_stages} stages for {well_name}")
        
        # Generate distinct colors for all stages
        colors = generate_distinct_colors(num_stages)

        for i, stage in enumerate(unique_stages):
            stage_data = combined_data[combined_data['stage'] == stage]
            logging.info(f"Stage {stage}: {len(stage_data)} data points")
            fig.add_trace(go.Scatter(
                x=stage_data['normalized_time'],
                y=stage_data['rolling_IR'],
                mode='lines',
                name=f'Stage {stage}',
                line=dict(color=colors[i], width=2),
                legendgroup=f'group{i}',
                showlegend=True
            ))
        
        # Calculate a suitable height for the plot
        plot_height = max(600, 400 + (num_stages * 20))  # Base height + extra height for each stage

        fig.update_layout(
            title=f'Rolling IR for {well_name} ({num_stages} stages)',
            xaxis_title='Time (seconds)',
            yaxis_title='Rolling IR',
            xaxis_range=[0, 900],
            legend_title="Stages",
            hovermode="x unified",
            height=plot_height,  # Set the calculated height
            legend=dict(
                itemsizing='constant',
                font=dict(size=10),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            margin=dict(r=150)  # Increase right margin to accommodate legend
        )
        fig.update_yaxes(exponentformat='e', showexponent='all')
        
        return fig
    except Exception as e:
        logging.error(f"Error in plot_rolling_ir: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise
