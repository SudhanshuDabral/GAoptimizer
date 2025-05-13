# Hydraulic Fracturing Productivity Model Optimizer (HF-PMO)

## Overview

The HF-PMO is a sophisticated genetic algorithm (GA)-based optimization tool designed for creating regression models that predict productivity in hydraulic fracturing operations. What sets this tool apart is its ability to enforce physically realistic monotonic relationships between key hydraulic fracturing parameters and productivity outcomes.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Features](#features)
- [Genetic Algorithm Process](#genetic-algorithm-process)
- [Monotonicity Enforcement](#monotonicity-enforcement)
- [Parameter Guide](#parameter-guide)
- [Usage Examples](#usage-examples)

## Introduction

In hydraulic fracturing, certain physical relationships should be maintained in any predictive model. For example, as parameters like downhole proppant per minute (downhole_ppm), total downhole proppant per minute (total_dhppm), and treating energy equivalent (tee) increase, we expect productivity to increase as well. Traditional regression techniques don't enforce these physical constraints, potentially leading to models that violate real-world physics.

The HF-PMO uses genetic algorithms to overcome this limitation, creating models that not only fit the data well (high R² values) but also respect the physical relationships between parameters.

## Setup

### Requirements

- Python 3.7+
- Streamlit
- Pandas, NumPy, Scikit-learn
- DEAP (Distributed Evolutionary Algorithms in Python)
- Plotly (for visualization)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features

- **Data Management**: Import from database or upload files
- **Z-Score Normalization**: Normalize data for better optimization
- **Custom Equation Support**: Evaluate existing equations
- **Monotonicity Enforcement**: Ensure physically realistic relationships
- **Parallel Processing**: Efficiently calculate ranges across multiple stages
- **Visual Analytics**: Track optimization progress in real-time
- **Model Export**: Save, download, and share optimized models

## Genetic Algorithm Process

The GA optimization in HF-PMO follows these key steps:

### 1. Initialization

- **Population Creation**: Generate an initial population of potential regression models
- **Feature Encoding**: Each individual represents a subset of features and their coefficients
- **Random Start**: Begin with randomly selected features and coefficient values

### 2. Fitness Evaluation

The fitness function balances multiple objectives:

- **R² Performance**: How well the model fits training and test data
- **Coefficient Constraints**: Ensuring coefficients stay within reasonable ranges
- **Monotonicity Score**: How well the model respects physical relationships
- **Weighted Metrics**: Train/test performance is weighted (30%/70%) to prevent overfitting

### 3. Selection

- **Tournament Selection**: Models compete in small groups
- **Elitism**: Top performers are preserved across generations
- **Fitness-Proportional Chances**: Better models have higher probability of selection

### 4. Crossover and Mutation

- **Feature Crossover**: Exchange features between promising models
- **Coefficient Mutation**: Randomly alter coefficients to explore new solutions
- **Adaptive Rates**: Crossover and mutation probabilities can be tuned

### 5. Evaluation and Iteration

- **Continuous Improvement**: Models evolve over multiple generations
- **Progress Tracking**: R² values are tracked and displayed in real-time
- **Multiple Models**: Generate several models meeting your criteria

### 6. Result Analysis

- **Feature Importance**: Analyze which parameters matter most
- **Sensitivity Testing**: Understand how parameters influence predictions
- **Monotonicity Verification**: Confirm physical relationships are maintained

## Detailed GA Implementation

This section provides a deeper dive into the genetic algorithm implementation used in HF-PMO, explaining technical aspects of how the optimization process works.

### Individual Encoding

In HF-PMO, each "individual" in the population represents a potential regression model:

1. **Binary Feature Selection**:
   - Each individual contains a binary array where 1 indicates a feature is included and 0 indicates it's excluded
   - Example: `[1,0,1,0,1,1,0]` means features at positions 0, 2, 4, and 5 are used in the model

2. **Coefficient Association**:
   - Each selected feature gets assigned a coefficient through the LinearRegression model
   - These coefficients become part of the individual's "genetic material" that evolves over time

3. **Model Representation**:
   ```python
   class Individual(list):
       # Properties
       fitness = None      # Fitness score
       model = None        # LinearRegression model
       features = []       # Selected feature indices
       weighted_r2 = 0.0   # Combined train/test R² score
       train_r2 = 0.0      # Training data R² score
       test_r2 = 0.0       # Test data R² score
       monotonicity_percent = 0.0  # Monotonicity score
       key_attr_monotonicity = {}  # Detailed monotonicity data
   ```

### Fitness Function Deep Dive

The fitness evaluation is the core of the GA process:

```python
def evalModel(individual):
    # Extract selected features
    features = [i for i, bit in enumerate(individual) if bit == 1]
    
    # Early exit conditions
    if not features or len(features) < 2:
        return 0,
    
    # Create training and test datasets with selected features only
    X_train_sub = X_train[:, features]
    X_test_sub = X_test[:, features]
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train_sub, y_train)
    
    # Early termination for out-of-range coefficients
    coefficients = model.coef_
    if not all(coef_range[0] <= coef <= coef_range[1] for coef in coefficients):
        return 0,
    
    # Calculate R² scores
    train_predictions = model.predict(X_train_sub)
    train_score = r2_score(y_train, train_predictions)
    
    # Early termination for poor training score
    if train_score < r2_threshold * 0.8:
        return 0,
    
    test_predictions = model.predict(X_test_sub)
    test_score = r2_score(y_test, test_predictions)
    
    # Calculate weighted score (prioritize test performance)
    weighted_score = 0.3 * train_score + 0.7 * test_score
    
    # Check monotonicity only for promising models
    monotonicity_percent = check_monotonicity_percent(...)
    monotonicity_penalty = max(0, monotonicity_target - monotonicity_percent) * 0.5
    
    # Coefficient penalty
    penalty = sum([max(0, coef - coef_range[1]) + max(0, coef_range[0] - coef) 
                  for coef in coefficients])
    penalty_factor = 0.01
    
    # Calculate final score with penalties
    penalized_score = weighted_score - (penalty * penalty_factor) - monotonicity_penalty
    
    # Attach properties to the individual for later reference
    individual.model = model
    individual.features = features
    individual.weighted_r2 = weighted_score
    individual.train_r2 = train_score
    individual.test_r2 = test_score
    individual.monotonicity_percent = monotonicity_percent
    
    return (penalized_score,)  # Return as tuple for DEAP compatibility
```

### Early Termination Optimizations

To improve performance, HF-PMO implements several early termination checks:

1. **Feature Count**: Models with too few features are immediately rejected
2. **Coefficient Range**: If coefficients fall outside allowed ranges, reject without further evaluation
3. **Training Score**: If training R² is too low, no need to evaluate test performance
4. **Weighted Score**: Only perform expensive monotonicity checks on promising models

These optimizations significantly improve performance by avoiding unnecessary calculations for obviously poor models.

### Genetic Operators in Detail

#### Selection: Tournament Selection

```python
# Tournament selection with tournament size 3
toolbox.register("select", tools.selTournament, tournsize=3)
```

Tournament selection:
1. Randomly selects 3 individuals from the population
2. The best individual from this tournament advances
3. This process is repeated until enough individuals are selected
4. Provides selection pressure while maintaining diversity

#### Crossover: Two-Point Crossover

```python
toolbox.register("mate", tools.cxTwoPoint)
```

Two-point crossover:
1. Selects two random points in the parent individuals
2. Exchanges the segments between these points
3. Creates offspring with mixed genetic material from both parents
4. Allows the algorithm to combine successful feature subsets

#### Mutation: Bit-Flip Mutation

```python
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
```

Bit-flip mutation:
1. For each bit in the individual, flip it with 5% probability
2. Adds or removes features from the model
3. Maintains genetic diversity
4. Enables exploration of the solution space

### Evolution Process Details

The main evolutionary loop follows this process:

```python
# Main evolutionary loop
for gen in range(num_generations):
    # Create offspring through variation operators
    offspring = algorithms.varAnd(population, toolbox, prob_crossover, prob_mutation)
    
    # Evaluate all offspring
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    # Select individuals for the next generation (include parents in selection)
    population = toolbox.select(offspring + population, k=population_size)
    
    # Track progress
    best_ind = tools.selBest(population, 1)[0]
    if best_ind.weighted_r2 > best_seen_r2:
        best_seen_r2 = best_ind.weighted_r2
        # Update best model information
```

### Convergence Determination

The GA in HF-PMO doesn't use traditional convergence criteria but instead:

1. Runs for a fixed number of generations (specified by user)
2. Tracks the best R² score seen so far
3. Maintains multiple high-performing models
4. Allows the user to stop the process manually if satisfied with results

### Implementation with DEAP

The system uses the DEAP (Distributed Evolutionary Algorithms in Python) library:

```python
from deap import base, creator, tools, algorithms

# Create fitness and individual types
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# Create the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalModel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
```

### Performance Optimizations

Several strategies improve GA performance:

1. **Parallel Evaluation**: Fitness evaluations can be parallelized
2. **Caching Results**: Store fitness values to avoid recalculating
3. **Vectorized Operations**: Use NumPy for efficient calculations
4. **Progressive Complexity**: Start with simpler models and increase complexity
5. **Feature Pre-screening**: Eliminate clearly irrelevant features early

### Statistical Considerations

The GA process includes several statistical safeguards:

1. **Train/Test Split**: Data is divided to prevent overfitting
2. **Cross-Validation**: K-fold validation can be applied during fitness evaluation
3. **Weighted Metrics**: More weight on test performance than training
4. **R² Thresholds**: Minimum acceptable performance levels

These details provide insight into how HF-PMO balances the competing objectives of statistical fit and physical realism through a carefully engineered genetic algorithm implementation.

## Monotonicity Enforcement

The key innovation in HF-PMO is its ability to enforce monotonicity, ensuring that as certain parameters increase, productivity predictions also increase.

### Monotonicity Calculation Process

1. **Range Determination**:
   - Ranges for each attribute are calculated from:
     - Selected database wells and their stages (processed in parallel)
     - Manual input ranges
     - Dataset statistics

2. **Monotonicity Testing**:
   ```python
   # Pseudocode for monotonicity checking
   for each attribute in selected_monotonic_attributes:
       generate test_points across attribute range
       calculate predictions at each test point
       if predictions consistently increase as attribute increases:
           attribute is monotonic
   ```

3. **Monotonicity Scoring**:
   - Each attribute receives a monotonicity score (0-100%)
   - Key attributes receive higher weights in the overall score
   - The GA optimizes for models with high monotonicity scores

4. **Fitness Penalties**:
   - Models with low monotonicity receive substantial fitness penalties
   - Penalties scale with the degree of monotonicity violation
   - Extra penalties applied when key attributes show decreasing trends

### Parallel Processing for Efficiency

Monotonicity range calculation is computationally intensive, especially with multiple wells and stages. HF-PMO uses parallel processing:

```python
# Parallel processing of stages
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_to_stage = {
        executor.submit(
            process_stage_data, 
            well_id, 
            stage, 
            key_attributes, 
            statistics
        ): (well_id, stage) for well_id, stage in all_tasks
    }
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(future_to_stage):
        stage_results = future.result()
        # Update attribute ranges with stage data
```

This approach significantly reduces processing time when analyzing multiple wells and stages.

## Parameter Guide

### GA Optimizer Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| R² Threshold | Minimum acceptable R² value | 0.5-0.9 |
| Coefficient Range | Limits for model coefficients | (-20, 20) |
| Monotonicity Target | Required percentage of monotonic behavior | 75-100% |
| Crossover Probability | Chance of feature exchange between models | 0.7-0.9 |
| Mutation Probability | Chance of random coefficient changes | 0.1-0.3 |
| Number of Generations | Iterations for evolution | 20-100 |
| Population Size | Number of concurrent models | 30-100 |
| Number of Models | Final models to generate | 1-6 |
| Regression Type | Full polynomial or linear with interactions | - |

### Monotonicity Settings

- **Selected Attributes**: Parameters that must show increasing trends with productivity
- **Range Determination Method**: How attribute ranges are calculated
- **Well Selection**: Which wells to use for determining realistic ranges

## Usage Examples

### Basic Workflow

1. **Select Data Source**: Choose database wells or upload a file
2. **Z-Score Data**: Normalize data for better comparability
3. **Select Parameters**:
   - Choose attributes for monotonicity enforcement (typically downhole_ppm, total_dhppm, tee)
   - Set GA parameters (generation count, population size, etc.)
   - Define monotonicity target percentage
4. **Run Optimization**: Start the GA process and monitor progress
5. **Analyze Results**:
   - Review R² scores and monotonicity percentages
   - Examine feature importance and sensitivity
   - Verify monotonic relationships in the models

### Advanced Usage

#### Custom Monotonicity Ranges

For specialized scenarios, manually define ranges for monotonicity testing:

1. Select "Manual Range Input" method
2. Define min/max values for each attribute
3. Run optimization with custom ranges

#### Model Validation

Test monotonicity of existing models:

1. Open the Monotonicity Check modal
2. Select wells and stages to test
3. Choose an existing model or enter a custom equation
4. Run the check to visualize monotonic relationships

## Technical Details

### Key Files and Functions

- `ga_calculation.py`: Core genetic algorithm implementation
- `check_monotonicity.py`: Functions for monotonicity validation
- `ga_main.py`: Main application interface and workflow
- `process_stage_data()`: Parallel processing for ranges
- `check_monotonicity_percent()`: Calculates monotonicity score
- `check_key_attributes_monotonicity()`: Detailed key attribute analysis

### Monotonicity Enforcement in the Fitness Function

The critical component of monotonicity enforcement occurs in the fitness function:

```python
def evalModel(individual):
    # Setup and basic fitness calculation...
    
    # Calculate monotonicity score
    monotonicity_percent = check_monotonicity_percent(
        model, X_poly, feature_names, features,
        prioritize_key_attributes=True, 
        attribute_ranges=monotonicity_ranges,
        selected_monotonic_attributes=selected_monotonic_attributes
    )
    
    # Apply penalty for low monotonicity
    monotonicity_penalty = max(0, monotonicity_target - monotonicity_percent) * 0.5
    
    # Check key attributes specifically
    if weighted_score >= r2_threshold * 0.9:
        key_attr_results = check_key_attributes_monotonicity(...)
        key_monotonicity = avg(key_attr_results.values())
        key_monotonicity_penalty = max(0, monotonicity_target - key_monotonicity) * 0.7
    
    # Final score with penalties
    penalized_score = weighted_score - monotonicity_penalty - key_monotonicity_penalty
    
    return penalized_score,
```

This approach balances the competing objectives of:
1. High statistical fit (R² value)
2. Physically realistic monotonic relationships
3. Coefficient constraints

The result is models that both fit the data well and respect the physical principles of hydraulic fracturing.

### In Simple Words 
The GA process ensures monotonicity for selected attributes through these key steps:
## Monotonicity Measurement:
For each potential model, the system tests monotonicity by generating test points across the range of each selected attribute
It checks if productivity increases as each attribute increases
Calculates a monotonicity percentage (what % of test points show monotonic behavior)
## Fitness Function Penalties:
The GA uses a fitness function that includes both R² performance and monotonicity
If a model's monotonicity percentage falls below your target (the slider you set, default 90%), it applies penalties
Selected attributes receive higher weights/penalties than other attributes
A model with poor monotonicity gets a significantly reduced fitness score
## Evolution Process:
Models with better fitness scores (higher R², better monotonicity) have higher chances of being selected
Through crossover and mutation, the GA gradually evolves toward solutions that balance R² performance with monotonic behavior
Models that satisfy both criteria rise to the top
## Key Attribute Enforcement:
The system applies extra penalties when selected hydraulic fracturing attributes don't show the physically expected increasing monotonicity
This steers the algorithm toward models where productivity increases with increases in critical parameters like downhole_ppm, tee, etc.
Rather than strictly forcing monotonicity (which might sacrifice fit), the GA balances model accuracy with physical realism by using these penalties to guide the evolution process toward models that respect the expected physical relationships.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
