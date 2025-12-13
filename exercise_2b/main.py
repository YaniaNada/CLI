import pandas as pd
from exercise_2b.model_training import call_model, call_predict, call_evaluate
from exercise_2b import split_dataset

# Load dataset
data = pd.read_excel('exercise_2b\dataset.csv')
df = pd.DataFrame(data)

# Identify the features (X) and target (y)
X = df[['beds', 'baths', 'size', 'zip_code']]
y = df['price']

X_train, X_test, y_train, y_test = split_dataset(X, y)

def main():
    call_model(X_train, y_train)
    y_pred = call_predict(X_test)
    result = call_evaluate(y_test, y_pred)

    # Display result
    print(f' house characteristics:/n {X_test}')
    print(f'/n Predicted house price: {y_pred}')
    print(f'/n Mean square error: {result.mse}')
    print(f'/n R-squared Score: {result.r2}')

if __name__ == '__main__':
    main()