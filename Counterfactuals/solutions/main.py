import numpy as np
import pandas as pd

from counterfactuals_solution import get_counterfactuals
from utils import load_data, encode_features, get_train_test_data, train_model, evaluate_model, generate_individual 

np.random.seed(42)

if __name__ == "__main__":
    # Load the data
    df, cfg = load_data(data_filepath='LoanApprovalPrediction.csv')
    df = encode_features(df, cfg["categorical"]) 
    
    # Split and train a model
    X_train, X_test, y_train, y_test = get_train_test_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Find a rejected individual among the model predictions
    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])
    i_denied = predictions_df.where(predictions_df.eq(0)).first_valid_index() 

    # Find the corresponding data point 
    x = X_test.loc[i_denied:i_denied]

    # Take out this data point from the test data to use the rest as X_obs
    X_obs = X_test.drop(i_denied)

    # Make a list of Feature objects containing information about how 
    # each feature is allowed to change when generating counterfactuals
    change_features = generate_individual(X_obs, x, cfg["feature_info"])
    
    # Set the desired new model prediction
    y_CF = 0.7
    print(f"Searching for counterfactuals with y_CF = {y_CF}...\n")
    numerical_features = [x for x in df.columns if x not in cfg["categorical"]]
    CFS = get_counterfactuals(X_obs, x, y_CF, model, 
                              numerical_features, 
                              cfg["categorical_features"], 
                              change_features, 
                              tol=0.05, 
                              optimization_steps=100, 
                              timeout=300) 

    # print(df.to_markdown(index=False))  # requires 'tabulate' installed
    print(f"Input point:\n{x.to_string(index=False)}")
    print(f"\nCounterfactuals:\n{CFS.to_string(index=False)}")
    

