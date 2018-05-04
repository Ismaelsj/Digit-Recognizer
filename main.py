import os.path
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import pandas as pd
import data_process
import model
import accuracy_estimation

def main():
        # Get the data
    print("Extracting data ...")
    df_train = pd.DataFrame(pd.read_csv("./train.csv"))
    df_test = pd.read_csv("./test.csv")

        # Split features train / test
    X_train, Y_train, X_test, Y_test = data_process.split_data(df_train)

        # Vectorize labels
    Y_train, Y_test = data_process.encode_labels(Y_train, Y_test)

        # Model and training parameters
    parameters = {}
    _, parameters['n_features'] = X_train.shape
    parameters['model_path'] = 'model/img_perceptron.ckpt'
    parameters['n_hidden'] = 2
    parameters['hidden_dim'] = 200
    parameters['n_class'] = 10
    parameters['learning_rate'] = 0.01
    parameters['training_epochs'] = 150
    parameters['visualize'] = False
    if ((len(argv) > 1 and argv[1] == "-v") or (len(argv) > 2 and argv[2] == "-v") or (len(argv) > 3 and argv[3] == "-v")):
        parameters['visualize'] = True

        # Make model
    nn_model = model.make_model(parameters)

        # Train model
    if ((len(argv) > 1 and argv[1] == "-n") or (len(argv) > 2 and argv[2] == "-n") or (len(argv) > 3 and argv[3] == "-n")):
        model.neural_network(X_train, Y_train, parameters, nn_model, X_test, Y_test)

        # Get accuracy
    if os.path.isfile(parameters['model_path']) == True:
        accuracy_estimation.Accuracy(parameters, nn_model, X_train, Y_train, X_test, Y_test)
        # Write estimation to "./estimation.csv"
    if os.path.isfile(parameters['model_path']) == True:
        accuracy_estimation.Estimation(parameters, nn_model, df_test)
    else:
        print("\nNo model found, please create a new file named 'img_perceptron.ckpt' in a directory named 'model' and launch the programme with the folowing commande :\n'python3 main.py -n'\n")

    if ((len(argv) > 1 and argv[1] == "-e") or (len(argv) > 2 and argv[2] == "-e") or (len(argv) > 3 and argv[3] == "-e")):
        accuracy_estimation.Exemple(df_test)

if __name__ == '__main__':
    main()
