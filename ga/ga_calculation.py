import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from deap import base, creator, tools, algorithms
import warnings
import streamlit as st
import time
from utils import plotting
import logging
from logging.handlers import RotatingFileHandler
import os

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
    logger.log(level, f"[GA Calculation] {message}")

# Move DEAP creator setup outside of run_ga function
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

def check_key_attributes_monotonicity(model, X, feature_names, selected_features, attribute_ranges=None, selected_monotonic_attributes=None):
    """
    Check monotonicity specifically for the selected attributes.
    Default key hydraulic fracturing attributes if none selected:
    - downhole_ppm
    - total_dhppm
    - tee
    
    Returns: dictionary with monotonicity scores for each key attribute,
    specifically checking for INCREASING monotonicity (productivity should
    increase as these attributes increase)
    """
    try:
        # Reduce number of test points to improve performance
        n_points = 30  # Reduced from 100 to 30
        
        # Default key attributes if none selected
        default_key_attributes = ['downhole_ppm', 'total_dhppm', 'tee']
        
        # Use selected attributes if provided, otherwise use defaults
        key_attributes = selected_monotonic_attributes if selected_monotonic_attributes else default_key_attributes
        
        results = {}
        
        # Get feature indices
        feature_indices = selected_features
        
        # For each feature, check monotonicity
        for feat_idx in feature_indices:
            feature_name = feature_names[feat_idx]
            
            # Only check selected attributes
            base_feature = feature_name.split()[0] if ' ' in feature_name else feature_name
            if base_feature not in key_attributes:
                continue
                
            # Get min and max values for the feature
            if attribute_ranges and base_feature in attribute_ranges:
                feat_min = attribute_ranges[base_feature]['min']
                feat_max = attribute_ranges[base_feature]['max']
            else:
                feat_min = X[:, feat_idx].min()
                feat_max = X[:, feat_idx].max()
            
            # Generate test points
            test_points = np.linspace(feat_min, feat_max, n_points)
            
            # For each point pair, check if productivity changes monotonically
            reference_point = X.mean(axis=0)
            
            # Vectorize predictions when possible
            test_matrix = np.tile(reference_point, (n_points, 1))
            test_matrix[:, feat_idx] = test_points
            
            # Filter to include only selected features
            test_matrix_filtered = test_matrix[:, feature_indices]
            
            # Get predictions all at once
            predictions = model.predict(test_matrix_filtered)
            
            # Check monotonicity
            monotonic_increases = np.sum(np.diff(predictions) >= 0)
            monotonic_strict_increases = np.sum(np.diff(predictions) > 0)
            non_monotonic_count = np.sum(np.diff(predictions) < 0)
            
            # Calculate percentages
            tests = n_points - 1  # Number of point pairs
            monotonic_percent = monotonic_increases / tests
            strict_monotonic_percent = monotonic_strict_increases / tests
            
            # We only care about INCREASING monotonicity for these key attributes
            results[feature_name] = {
                'monotonic_percent': monotonic_percent,
                'strict_monotonic_percent': strict_monotonic_percent,
                'direction': "increasing" if monotonic_strict_increases > (tests / 2) else "not consistently increasing",
                'test_points': test_points,
                'predictions': predictions,
                'non_monotonic_count': non_monotonic_count
            }
        
        return results
    
    except Exception as e:
        log_message(logging.WARNING, f"Error in key attributes monotonicity check: {str(e)}")
        return {}

def check_monotonicity_percent(model, X, feature_names, selected_features, prioritize_key_attributes=True, attribute_ranges=None, selected_monotonic_attributes=None):
    """
    Check what percentage of the feature space exhibits monotonic behavior
    for the given model with respect to each feature.
    
    For selected attributes, we specifically check for INCREASING monotonicity.
    
    Returns: float between 0 and 1 representing the overall monotonicity percentage
    """
    try:
        # Reduce number of test points to improve performance
        n_points = 20  # Further reduced from 30 to 20
        monotonic_count = 0
        total_tests = 0
        
        # Default key attributes to prioritize if enabled
        default_key_attributes = ['downhole_ppm', 'total_dhppm', 'tee']
        
        # Use selected attributes if provided, otherwise use defaults
        key_attributes = selected_monotonic_attributes if selected_monotonic_attributes else default_key_attributes
        
        key_attribute_weight = 2.5  # Increased weight for key attributes
        
        # Get feature indices
        feature_indices = selected_features
        
        # Process features in two groups: key attributes first, then other features if needed
        features_to_check = []
        
        # First, add all key attributes that exist in the feature set
        for i, feat_idx in enumerate(feature_indices):
            feature_name = feature_names[feat_idx]
            
            # Skip interaction terms for monotonicity check (they contain spaces)
            if ' ' in feature_name:
                continue
                
            # Determine if this is a key attribute
            base_feature = feature_name.split()[0] if ' ' in feature_name else feature_name
            is_key_attribute = base_feature in key_attributes
            
            if is_key_attribute:
                features_to_check.append((feat_idx, feature_name, base_feature, is_key_attribute))
        
        # Optional: add a limited number of other features if requested
        if not prioritize_key_attributes:
            for i, feat_idx in enumerate(feature_indices):
                feature_name = feature_names[feat_idx]
                
                # Skip interaction terms
                if ' ' in feature_name:
                    continue
                    
                # Skip key attributes as they're already added
                base_feature = feature_name.split()[0] if ' ' in feature_name else feature_name
                is_key_attribute = base_feature in key_attributes
                
                if not is_key_attribute:
                    features_to_check.append((feat_idx, feature_name, base_feature, is_key_attribute))
        
        # Only check a limited number of non-key attributes to save time
        max_non_key_features = 5
        non_key_features = [f for f in features_to_check if not f[3]]
        if len(non_key_features) > max_non_key_features:
            # Keep only the first few non-key features
            non_key_features = non_key_features[:max_non_key_features]
            # Replace non-key features with the limited set
            features_to_check = [f for f in features_to_check if f[3]] + non_key_features
        
        # Check monotonicity for each selected feature
        for feat_idx, feature_name, base_feature, is_key_attribute in features_to_check:
            # Get min and max values for the feature
            if attribute_ranges and base_feature in attribute_ranges:
                feat_min = attribute_ranges[base_feature]['min']
                feat_max = attribute_ranges[base_feature]['max']
            else:
                feat_min = X[:, feat_idx].min()
                feat_max = X[:, feat_idx].max()
            
            # Generate test points
            test_points = np.linspace(feat_min, feat_max, n_points)
            
            # For each point pair, check if productivity changes monotonically
            reference_point = X.mean(axis=0)
            
            # Vectorize predictions when possible
            test_matrix = np.tile(reference_point, (n_points, 1))
            test_matrix[:, feat_idx] = test_points
            
            # Filter to include only selected features
            test_matrix_filtered = test_matrix[:, feature_indices]
            
            # Get predictions all at once
            predictions = model.predict(test_matrix_filtered)
            
            # Calculate monotonicity using vectorized operations
            increases = np.diff(predictions) >= 0
            decreases = np.diff(predictions) <= 0
            
            monotonic_increases = np.sum(increases)
            monotonic_decreases = np.sum(decreases)
            
            # For key attributes, we specifically want INCREASING monotonicity
            if is_key_attribute:
                monotonic_percent = monotonic_increases / (n_points - 1)
            else:
                # For other attributes, we accept either increasing or decreasing
                monotonic_percent = max(monotonic_increases, monotonic_decreases) / (n_points - 1)
            
            feature_weight = key_attribute_weight if prioritize_key_attributes and is_key_attribute else 1.0
            
            if monotonic_percent >= 0.9:  # Allow 90% monotonicity for this feature
                monotonic_count += feature_weight
            
            total_tests += feature_weight
        
        # Return overall monotonicity percentage across all features
        if total_tests == 0:
            return 0.0
        return monotonic_count / total_tests
    
    except Exception as e:
        log_message(logging.WARNING, f"Error in monotonicity check: {str(e)}")
        return 0.0

def run_ga(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, timer_placeholder, regression_type, model_number, r2_values, iterations, model_markers, plot_placeholder, start_iteration, monotonicity_target=0.9, monotonicity_ranges=None, selected_monotonic_attributes=None):
    log_message(logging.INFO, f"Starting GA optimization for Model {model_number + 1}")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        try:
            # Remove data_id and well_id from predictors if they exist
            predictors = [p for p in predictors if p not in ['data_id', 'well_id']]
            
            X = df[predictors].values
            y = df[target_column].values

            # Create an empty placeholder for the plot if not already present
            if plot_placeholder.empty:
                plot_placeholder.empty()

            # Show optimization status message
            status_message = st.empty()
            status_message.text(f"Optimizing model {model_number + 1}... Setting up data and model")
            
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(df[predictors].columns)

            if regression_type == 'LWIP':
                quadratic_indices = [i for i, name in enumerate(feature_names) if '^2' in name]
                X_poly = np.delete(X_poly, quadratic_indices, axis=1)
                feature_names = [name for i, name in enumerate(feature_names) if i not in quadratic_indices]

            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            toolbox = base.Toolbox()
            toolbox.register("attr_bool", np.random.randint, 0, 2)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evalModel(individual):
                features = [i for i, bit in enumerate(individual) if bit == 1]
                if not features or len(features) < 2:  # Require at least 2 features
                    return 0,
                
                try:
                    X_train_sub = X_train[:, features]
                    X_test_sub = X_test[:, features]
                except IndexError:
                    log_message(logging.WARNING, "IndexError in feature selection")
                    return 0,
                
                model = LinearRegression()
                model.fit(X_train_sub, y_train)
                
                # Get coefficients and check if they're within range early
                coefficients = model.coef_
                if not all(coef_range[0] <= coef <= coef_range[1] for coef in coefficients):
                    # Early termination if coefficients are out of range
                    return 0,
                
                train_predictions = model.predict(X_train_sub)
                train_score = r2_score(y_train, train_predictions)
                
                # Early termination if train score is too low
                if train_score < r2_threshold * 0.8:  # Allow some leeway
                    return 0,
                
                test_predictions = model.predict(X_test_sub)
                test_score = r2_score(y_test, test_predictions)
                
                weighted_score = 0.3 * train_score + 0.7 * test_score
                
                # Early termination if weighted score is too low
                if weighted_score < r2_threshold * 0.7:
                    return 0,
                
                # Only check monotonicity for promising models to save time
                monotonicity_percent = check_monotonicity_percent(model, X_poly, feature_names, features, 
                                                               prioritize_key_attributes=True, 
                                                               attribute_ranges=monotonicity_ranges,
                                                               selected_monotonic_attributes=selected_monotonic_attributes)
                
                monotonicity_penalty = max(0, monotonicity_target - monotonicity_percent) * 0.5
                
                # Only check key attributes monotonicity for very promising models
                if weighted_score >= r2_threshold * 0.9:
                    key_attr_results = check_key_attributes_monotonicity(model, X_poly, feature_names, features,
                                                                     attribute_ranges=monotonicity_ranges,
                                                                     selected_monotonic_attributes=selected_monotonic_attributes)
                    
                    # Calculate average monotonicity of key attributes
                    key_monotonicity = 0.0
                    if key_attr_results:
                        key_monotonicity = sum(result['monotonic_percent'] for result in key_attr_results.values()) / len(key_attr_results)
                    
                    # Apply extra penalty for low key attribute monotonicity
                    key_monotonicity_penalty = max(0, monotonicity_target - key_monotonicity) * 0.7
                else:
                    key_attr_results = {}
                    key_monotonicity_penalty = 0.0
                
                # Calculate final penalized score
                penalty = sum([max(0, coef - coef_range[1]) + max(0, coef_range[0] - coef) for coef in coefficients])
                penalty_factor = 0.01
                penalized_score = weighted_score - (penalty * penalty_factor) - monotonicity_penalty - key_monotonicity_penalty

                individual.train_r2 = train_score
                individual.test_r2 = test_score
                individual.weighted_r2 = weighted_score
                individual.model = model
                individual.features = features
                individual.monotonicity_percent = monotonicity_percent
                individual.key_attr_monotonicity = key_attr_results

                return penalized_score,

            toolbox.register("evaluate", evalModel)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)

            best_weighted_r2_score = 0
            best_model = None
            best_equation = None
            best_selected_features = None
            best_errors_df = None
            valid_coefficients = False
            iteration = start_iteration  # Start from the provided iteration number

            equation_placeholder = st.empty()
            selected_features_placeholder = st.empty()

            start_time = time.time()
            
            # Create progress indicator for generations
            progress_bar = st.progress(0)
            last_plot_update = time.time()
            update_frequency = 2.0  # Update plot every 2 seconds

            while (best_weighted_r2_score < r2_threshold or not valid_coefficients) and st.session_state.ga_optimizer['running']:
                iteration += 1
                pop = toolbox.population(n=population_size)
                
                # Run evolution with progress updates
                for gen in range(num_generations):
                    # Update progress bar
                    progress = (gen + 1) / num_generations
                    progress_bar.progress(progress)
                    
                    # Update status message
                    status_message.text(f"Optimizing model {model_number + 1}... Generation {gen + 1}/{num_generations} (R² so far: {best_weighted_r2_score:.4f})")
                    
                    # Evolve population for one generation
                    offspring = algorithms.varAnd(pop, toolbox, prob_crossover, prob_mutation)
                    fits = toolbox.map(toolbox.evaluate, offspring)
                    for fit, ind in zip(fits, offspring):
                        ind.fitness.values = fit
                    pop = toolbox.select(offspring + pop, k=population_size)
                    
                    # Find best individual in this generation
                    gen_best = tools.selBest(pop, 1)[0]
                    if hasattr(gen_best, 'weighted_r2') and gen_best.weighted_r2 > best_weighted_r2_score:
                        best_weighted_r2_score = gen_best.weighted_r2
                        best_model = gen_best
                        
                        # Update R2 plot periodically to avoid excessive refreshes
                        current_time = time.time()
                        if current_time - last_plot_update > update_frequency:
                            r2_values.append(best_weighted_r2_score)
                            iterations.append(iteration)
                            plotting.update_plot(iterations, r2_values, plot_placeholder, model_markers)
                            last_plot_update = current_time
                
                # Always update after full evolution cycle
                r2_values.append(best_weighted_r2_score)
                iterations.append(iteration)
                plotting.update_plot(iterations, r2_values, plot_placeholder, model_markers)
                
                if best_model is not None:
                    selected_features = best_model.features
                    selected_feature_names = [feature_names[i] for i in selected_features]
                    
                    model = best_model.model
                    coefficients = model.coef_
                    intercept = model.intercept_
                    
                    valid_coefficients = all(coef_range[0] <= coef <= coef_range[1] for coef in coefficients)
                    
                    if valid_coefficients:
                        equation = f"Corrected_Prod = {intercept:.4f}"
                        selected_feature_names = []
                        for coef, feature in zip(coefficients, [feature_names[i] for i in selected_features]):
                            terms = feature.split()
                            if len(terms) == 1:
                                equation += f" + ({coef:.4f} * {terms[0]})"
                                selected_feature_names.append(terms[0])
                            elif len(terms) == 2:
                                equation += f" + ({coef:.4f} * {terms[0]} * {terms[1]})"
                                selected_feature_names.append(f"{terms[0]} * {terms[1]}")
                            elif len(terms) == 3 and terms[1] == terms[2]:
                                equation += f" + ({coef:.4f} * {terms[0]} * {terms[0]})"
                                selected_feature_names.append(f"{terms[0]}^2")
                            else:
                                equation += f" + ({coef:.4f} * {' * '.join(terms)})"
                                selected_feature_names.append(' * '.join(terms))
                        
                        X_sub = X_poly[:, selected_features]
                        y_pred = model.predict(X_sub)
                        errors = (y - y_pred)**2
                        
                        # Create errors_df with available columns
                        errors_df_data = {
                            'WellName': df['Well Name'],
                            'stage': df['stage'],
                            'Actual': y,
                            'Predicted': y_pred,
                            'Error': errors
                        }
                        
                        # Add data_id and well_id if they exist
                        if 'data_id' in df.columns:
                            errors_df_data['data_id'] = df['data_id']
                        if 'well_id' in df.columns:
                            errors_df_data['well_id'] = df['well_id']
                            
                        errors_df = pd.DataFrame(errors_df_data)

                        best_equation = equation
                        best_selected_features = selected_feature_names
                        best_errors_df = errors_df

                        log_message(logging.INFO, f"New best model found for Model {model_number + 1} (R² score: {best_weighted_r2_score:.4f}, Monotonicity: {best_model.monotonicity_percent:.2f})")
                        
                elapsed_time = time.time() - start_time
                timer_placeholder.write(f"Time Elapsed: {elapsed_time:.2f} seconds")

                if not st.session_state.ga_optimizer['running']:
                    log_message(logging.INFO, f"GA optimization stopped by user for Model {model_number + 1}")
                    break

            # When a valid model is found, add it to model_markers
            if valid_coefficients:
                model_markers[model_number + 1] = (iteration, best_weighted_r2_score)
                plotting.update_plot(iterations, r2_values, plot_placeholder, model_markers)
                status_message.text(f"Model {model_number + 1} optimization complete! (R²: {best_weighted_r2_score:.4f})")
            else:
                status_message.text(f"Could not find a valid model {model_number + 1}. Please try again with different parameters.")

            # Clean up progress indicators
            progress_bar.empty()

            # Calculate R² score on entire dataset
            if best_model is not None:
                X_full = X_poly[:, best_model.features]
                y_pred_full = best_model.model.predict(X_full)
                full_dataset_r2 = r2_score(y, y_pred_full)
                log_message(logging.INFO, f"GA optimization completed for Model {model_number + 1}. Full dataset R²: {full_dataset_r2:.4f}, Monotonicity: {best_model.monotonicity_percent:.2f}")
            else:
                full_dataset_r2 = 0
                log_message(logging.WARNING, f"GA optimization completed for Model {model_number + 1}. No valid model found")

        except Exception as e:
            log_message(logging.ERROR, f"Error during GA optimization for Model {model_number + 1}: {str(e)}")
            return None

        if caught_warnings:
            for warn in caught_warnings:
                log_message(logging.WARNING, f"Warning during GA run for Model {model_number + 1}: {warn.message}")

    if not st.session_state.ga_optimizer['running']:
        return None

    return best_model, best_weighted_r2_score, best_equation, best_selected_features, best_errors_df, full_dataset_r2, iteration

if __name__ == "__main__":
    # This block can be used for testing the function independently
    pass
