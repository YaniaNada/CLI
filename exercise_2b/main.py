# use argument passer to take in some feature values from command line 
# and predict the price for that specific house
# then save the prediction to a text file

import pandas as pd
import argparse
import exercise_2b.model_training as model_train
from exercise_1a.write_function import write_fn

parser = argparse.ArgumentParser(description="Predict house price based on features")
parser.add_argument('--beds', type = int, required = True, help = 'number of beds')
parser.add_argument('--baths', type = int, required = True, help = 'number of baths')
parser.add_argument('-s', '--size', type = float, required = True, help = 'size of the house in square feet')
parser.add_argument('-z', '--zip_code', type = int, required = True, help = 'zip code of the house location')

args = parser.parse_args()

def main():

    features = pd.DataFrame([[args.beds, args.baths, args.size, args.zip_code]])
    features.columns = ['beds', 'baths', 'size', 'zip_code']

    predicted_value = model_train.ModelObject.call_predict(features)
    predicted_value = (predicted_value[0]).round(3)
    mse, r2 = model_train.mse, model_train.r2

    
    # Write result to a file
    filepath = 'exercise_2b/predicted_price.txt'
    output = f'Property feature: \n{features}, \nPredicted price: {predicted_value}'
    write_fn(output, filepath) 

if __name__ == '__main__':
    main()

# provide features in this format when running the script: --beds 2 --baths 2 --size 900 -z 98107