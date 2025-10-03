import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler

from utils import epsilon_rounding, get_relevant_candidates

optuna.logging.set_verbosity(optuna.logging.WARNING)

def misfit(x_prime, y_target, model):
    """
    Optimisation criterion 1
    Calculate absolute difference between y_target and y_prime_prediction.
    """
    y_prime_prediction = model.predict_proba(x_prime)[0][0]
    return abs(y_target - y_prime_prediction)

def distance(X, x, x_prime, numerical, categorical):
    """
    Optimisation criterion 2
    Calculate distance between x_prime and x.
    """
    # Normalize data
    scaler = MinMaxScaler()
    scaler.fit(X[numerical])
    x_normalized = scaler.transform(x[numerical])
    x_prime_normalized = scaler.transform(x_prime[numerical])
        
    # Compute distances
    numeric_distance = np.linalg.norm(x_prime_normalized - x_normalized, axis=1)
    categorical_distance = (x_prime[categorical].values != x[categorical].values).sum(axis=1)
    
    return numeric_distance + categorical_distance

def sparsity(x, x_prime):
    """
    Optimisation criterion 3
    Return number of unchanged features.
    """
    return (x.values[0] == x_prime.values[0]).sum() / len(x.values[0])

def closest_real(X, x_prime, categorical, numerical):
    """
    Optimisation criterion 4
    Return the minimum distance between x_prime and any point in X.
    """
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X[numerical])
    x_prime_normalized = scaler.transform(x_prime[numerical])

    numeric_distance = np.linalg.norm(x_prime_normalized - X_normalized, axis=1)
    categorical_distance = (x_prime[categorical].values != X[categorical].values).sum(axis=1)
    
    # Compute total distance
    total_distance = numeric_distance + categorical_distance
    
    return min(total_distance)


def objective(trial, X, x, features, model, y_target, numerical, categorical):
    x_prime = x.copy()
    for feature in features:
        feature.sample(trial)
        x_prime[feature.name] = feature.value
    epsilon_rounding(x, x_prime, 1e-1)

    obj1 = misfit(x_prime, y_target, model)
    obj2 = distance(X, x, x_prime, numerical, categorical)
    obj3 = sparsity(x, x_prime)
    obj4 = closest_real(X, x_prime, categorical, numerical)

    return obj1, obj2, obj3, obj4

def get_counterfactuals(X, x, y_target, model, 
                        numerical, categorical, features, 
                        tol, optimization_steps, timeout):

    study = optuna.create_study(directions=['minimize', 'minimize', 'maximize', 'minimize'], 
                                sampler=optuna.samplers.NSGAIISampler(seed=42)) 
    
    study.optimize(lambda trial: objective(trial, X, x, features, model, 
                                           y_target, 
                                           numerical, 
                                           categorical), 
                   n_trials=optimization_steps, 
                   timeout=timeout)
    
    candidates_df = get_relevant_candidates(study, x, model, y_target, tol)
    
    return candidates_df
