import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from deap import base, creator, tools, algorithms
import warnings
import streamlit as st
import time
from utils import plotting

warnings.filterwarnings('ignore')

def run_ga(df, target_column, predictors, r2_threshold, coef_range, prob_crossover, prob_mutation, num_generations, population_size, timer_placeholder):
    X = df[predictors].values
    y = df[target_column].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(df[predictors].columns)

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
            return (0,)  # Avoid models with no features
        
        X_train_sub = X_train[:, features]
        X_test_sub = X_test[:, features]
        
        model = LinearRegression()
        model.fit(X_train_sub, y_train)
        
        predictions = model.predict(X_test_sub)
        score = r2_score(y_test, predictions)

        # Apply penalty for coefficients outside the desired range
        coefficients = model.coef_
        penalty = sum([max(0, coef - coef_range[1]) + max(0, coef_range[0] - coef) for coef in coefficients])
        penalty_factor = 0.01  # Adjust this factor to balance the penalty impact

        penalized_score = score - penalty * penalty_factor

        return (penalized_score,)

    toolbox.register("evaluate", evalModel)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    best_r2_score = 0
    valid_coefficients = False
    iteration = 0

    plot_placeholder = st.empty()
    equation_placeholder = st.empty()
    selected_features_placeholder = st.empty()

    start_time = time.time()

    # Initialize plot data
    r2_values = []
    iterations = []

    while (best_r2_score < r2_threshold or not valid_coefficients) and st.session_state.running:
        iteration += 1
        print(f"Iteration: {iteration}")
        pop = toolbox.population(n=population_size)
        
        algorithms.eaSimple(pop, toolbox, prob_crossover, prob_mutation, num_generations, verbose=True)
        
        best_ind = tools.selBest(pop, 1)[0]
        best_r2_score = best_ind.fitness.values[0]
        
        # Update plot data
        r2_values.append(best_r2_score)
        iterations.append(iteration)
        plotting.update_plot(iterations, r2_values, plot_placeholder)
        
        # Update timer
        elapsed_time = time.time() - start_time
        timer_placeholder.write(f"Time Elapsed: {elapsed_time:.2f} seconds")

        if best_r2_score >= r2_threshold:
            selected_features = [i for i, bit in enumerate(best_ind) if bit == 1]
            selected_feature_names = [poly_feature_names[i] for i in selected_features]
            
            X_train_sub = X_train[:, selected_features]
            X_test_sub = X_test[:, selected_features]
            
            model = LinearRegression()
            model.fit(X_train_sub, y_train)
            
            coefficients = model.coef_
            intercept = model.intercept_
            
            valid_coefficients = all(coef_range[0] <= coef <= coef_range[1] for coef in coefficients)
            
            if valid_coefficients:
                equation = f"Corrected_Prod = {intercept:.4f}"
                for coef, feature in zip(coefficients, selected_feature_names):
                    equation += f" + ({coef:.4f} * {feature})"
                
                results = {
                    'Selected Features': selected_feature_names,
                    'Coefficient': coefficients
                }
                results_df = pd.DataFrame(results)
                results_df.loc['Intercept'] = ['Intercept', intercept]
                results_df.loc['R² Value'] = ['R² Value', best_r2_score]
                results_df.loc['Equation'] = ['Equation', equation]
                
                output_file = 'genetic_algorithm_results.xlsx'
                results_df.to_excel(output_file, index=False)
                
                print("Best individual is %s, with R² score of %s" % (best_ind, best_r2_score))
                print("Response Equation:", equation)

                equation_placeholder.write(f"Response Equation: {equation}")
                selected_features_placeholder.write(f"Selected Features: {selected_feature_names}")

                # Calculate errors for individual data points
                y_pred = model.predict(X_poly[:, selected_features])
                errors = (y - y_pred)**2
                errors_df = pd.DataFrame({
                    'Actual': y,
                    'Predicted': y_pred,
                    'Error': errors
                })

                if not st.session_state.running:
                    return best_ind, best_r2_score, equation, selected_feature_names, errors_df

    if not st.session_state.running:
        return None
    return best_ind, best_r2_score, equation, selected_feature_names, errors_df
