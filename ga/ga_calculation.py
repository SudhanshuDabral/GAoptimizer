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

warnings.filterwarnings('ignore')

def run_ga(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, timer_placeholder, regression_type):
    predictors = [p for p in predictors if p not in ['data_id', 'well_id']]
    
    X = df[predictors].values
    y = df[target_column].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(df[predictors].columns)

    if regression_type == 'LWIP':
        quadratic_indices = [i for i, name in enumerate(feature_names) if '^2' in name]
        X_poly = np.delete(X_poly, quadratic_indices, axis=1)
        feature_names = [name for i, name in enumerate(feature_names) if i not in quadratic_indices]

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalModel(individual):
        features = [i for i, bit in enumerate(individual) if bit == 1]
        if not features:
            return 0,  # Avoid models with no features
        
        X_train_sub = X_train[:, features]
        X_test_sub = X_test[:, features]
        
        model = LinearRegression()
        model.fit(X_train_sub, y_train)
        
        train_predictions = model.predict(X_train_sub)
        test_predictions = model.predict(X_test_sub)
        
        train_score = r2_score(y_train, train_predictions)
        test_score = r2_score(y_test, test_predictions)
        
        weighted_score = 0.3 * train_score + 0.7 * test_score

        # Apply penalty for coefficients outside the desired range
        coefficients = model.coef_
        penalty = sum([max(0, coef - coef_range[1]) + max(0, coef_range[0] - coef) for coef in coefficients])
        penalty_factor = 0.01  # Adjust this factor to balance the penalty impact

        penalized_score = weighted_score - penalty * penalty_factor

        # Store scores and model in the individual's attributes
        individual.train_r2 = train_score
        individual.test_r2 = test_score
        individual.weighted_r2 = weighted_score
        individual.model = model
        individual.features = features

        return penalized_score,  # Return as a tuple

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
    iteration = 0

    plot_placeholder = st.empty()
    equation_placeholder = st.empty()
    selected_features_placeholder = st.empty()

    start_time = time.time()

    r2_values = []
    iterations = []

    best_weighted_r2_score = 0
    best_model = None
    best_equation = None
    best_selected_features = None
    best_errors_df = None
    valid_coefficients = False
    iteration = 0

    plot_placeholder = st.empty()
    equation_placeholder = st.empty()
    selected_features_placeholder = st.empty()

    start_time = time.time()

    r2_values = []
    iterations = []

    while (best_weighted_r2_score < r2_threshold or not valid_coefficients) and st.session_state.ga_optimizer['running']:
        iteration += 1
        pop = toolbox.population(n=population_size)
        
        algorithms.eaSimple(pop, toolbox, prob_crossover, prob_mutation, num_generations, verbose=True)
        
        best_ind = tools.selBest(pop, 1)[0]
        
        if best_ind.weighted_r2 > best_weighted_r2_score:
            best_weighted_r2_score = best_ind.weighted_r2
            best_model = best_ind

            selected_features = best_model.features
            selected_feature_names = [feature_names[i] for i in selected_features]
            
            model = best_model.model
            coefficients = model.coef_
            intercept = model.intercept_
            
            valid_coefficients = all(coef_range[0] <= coef <= coef_range[1] for coef in coefficients)
            
            if valid_coefficients:
                equation = f"Corrected_Prod = {intercept:.4f}"
                for coef, feature in zip(coefficients, selected_feature_names):
                    terms = feature.split()
                    if len(terms) == 1:
                        equation += f" + ({coef:.4f} * {terms[0]})"
                    elif len(terms) == 2:
                        equation += f" + ({coef:.4f} * {terms[0]} * {terms[1]})"
                    elif len(terms) == 3 and terms[1] == terms[2]:
                        equation += f" + ({coef:.4f} * {terms[0]} * {terms[0]})"
                    else:
                        equation += f" + ({coef:.4f} * {' * '.join(terms)})"
                
                X_sub = X_poly[:, selected_features]
                y_pred = model.predict(X_sub)
                errors = (y - y_pred)**2
                errors_df = pd.DataFrame({
                    'WellName': df['Well Name'],
                    'stage': df['stage'],
                    'Actual': y,
                    'Predicted': y_pred,
                    'Error': errors
                })

                best_equation = equation
                best_selected_features = selected_feature_names
                best_errors_df = errors_df

                equation_placeholder.write(f"Response Equation: {equation}")
                selected_features_placeholder.write(f"Selected Features: {selected_feature_names}")

        r2_values.append(best_weighted_r2_score)
        iterations.append(iteration)
        plotting.update_plot(iterations, r2_values, plot_placeholder)
        
        elapsed_time = time.time() - start_time
        timer_placeholder.write(f"Time Elapsed: {elapsed_time:.2f} seconds")

        if not st.session_state.ga_optimizer['running']:
            break

    # Calculate R² score on entire dataset
    if best_model is not None:
        X_full = X_poly[:, best_model.features]
        y_pred_full = best_model.model.predict(X_full)
        full_dataset_r2 = r2_score(y, y_pred_full)
    else:
        full_dataset_r2 = 0

    if not st.session_state.ga_optimizer['running']:
        return None

    return best_model, best_weighted_r2_score, best_equation, best_selected_features, best_errors_df, full_dataset_r2