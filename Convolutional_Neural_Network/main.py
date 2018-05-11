import os.path
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import pandas as pd
import data_process
import model
import accuracy_estimation

def main():
    if ((len(argv) > 1 and (argv[1] != "-v" and argv[1] != "-n" and argv[1] != "-e"))
    or (len(argv) > 2 and (argv[2] != "-v" and argv[2] != "-n" and argv[2] != "-e"))
    or (len(argv) > 3 and (argv[3] != "-v" and argv[3] != "-n" and argv[3] != "-e"))):
        print("\nUsage: python3 main.py [-n][-v][-e]\n\t- Use '-n' to train a new model.\n\t- Use '-v' to visualize cost.\n\t- Use '-e' to visualize an exemple of prediction.\n")
        return 0
        # Get the data
    print("Extracting data ...")
    df_train = pd.DataFrame(pd.read_csv("../train.csv"))
    df_test = pd.read_csv("../test.csv")

        # Split features train / test
    features = df_train.shape[1] - 1
    X_train, Y_train, X_test, Y_test = data_process.split_data(df_train)

        # Vectorize labels
    Y_train, Y_test = data_process.encode_labels(Y_train, Y_test)

        # Model and training parameters
    parameters = {}
    parameters['batches_size'] = 50
    parameters['n_features'] = features
    parameters['model_path'] = 'model/img_perceptron.ckpt'
    parameters['n_channels'] = 1
    parameters['padding'] = "SAME"
    parameters['n_class'] = 10
    parameters['learning_rate'] = 0.003
    parameters['training_epochs'] = 15
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
