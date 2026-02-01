import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def write_fn(output, file_path):
    '''
    Write output to a file
    param output: Result to be written to file
    param file_path: Path to the file where output is to be written
    '''
    with open(file_path,'a') as f:
        f.write(output+'\n')
