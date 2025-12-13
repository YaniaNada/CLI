from exercise_2b import model_selection as model
from exercise_2b import model_evaluation as evaluate


def call_model(X_train, y_train):
    '''Train the model'''
    model.fit(X_train, y_train)

def call_predict(X_test):
    '''Make prediction'''
    y_pred = model.predict(X_test)
    return y_pred

def call_evaluate(y_test, y_pred):
    '''Evaluate the model'''
    mse,r2 = evaluate(y_test, y_pred)
    return mse, r2

def main():
    pass

if __name__ == '__main__':
    main()