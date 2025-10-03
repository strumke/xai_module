from utils import load_data, encode_features, get_train_test_data, train_model

# Load the data
df, cfg = load_data(data_filepath='Loan_data_train.csv')
df = encode_features(df, cfg["categorical"]) 
    
# Split and train a model
X_train, X_test, y_train, y_test = get_train_test_data(df)
model = train_model(X_train, y_train)
model.save_model("xgboost_model.json")
