import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def model_selection(model_type, param_grid):
    '''
    Docstring for model_selection
    
    :param model_type: Description
    '''
    model = model_type
    param_grid = model.param_grid
    return model, param_grid


def grid_search(model, param_grid, Xtrain, ytrain):
    '''
    Perform grid search to find the best model
    param model: The model to be tuned
    param param_grid: The hyperparameters to be tuned
    param Xtrain: Training data features
    param ytrain: Training data labels
    return: The best estimator from the grid search
    '''        
    grid = GridSearchCV(model, param_grid=param_grid, scoring= 'accuracy')
    grid.fit(Xtrain, ytrain)

    return grid.best_estimator_

def classification_report_fn(ytest, yfit, target_names):
    ''' 
    Print the classification report
    param ytest: True labels of the test data
    param yfit: Predicted labels of the test data
    param faces: Dataset containing the target names'''
    output = classification_report(ytest, yfit, target_names=target_names)
    print(output)
    return output

def plot_confusion_matrix(ytest, yfit, target_names, model_name):
    '''
    Plot the confusion matrix
    param ytest: True labels of the test data
    param yfit: Predicted labels of the test data
    param faces: Dataset containing the target names
    '''
    matrix = confusion_matrix(ytest, yfit)
    sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar='False', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    plt.savefig(f'exercise_4/training_images/confusion_matrix_{model_name}.png')
    print(f"Confusion matrix saved as confusion_matrix_{model_name}.png")
    return np.array2string(matrix)

def display_images(Xtest, ytest, yfit, target_names):
    ''' 
    Plot the test images with predicted and true labels
    param Xtest: Test data containing the images'''

    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(Xtest[i].reshape(62,47), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(target_names[yfit[i]].split()[-1], color='black' if yfit[i]==ytest[i] else 'red')

    fig.suptitle('Predicted Names; Incorrect labels in Red', size=14)

def write_fn(output, file_path):
    '''
    Write output to a file
    param output: Result to be written to file
    param file_path: Path to the file where output is to be written
    '''
    with open(file_path,'a') as f:
        f.write(output+'\n')


