import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def make_model(parameters):
    # Network parmaeters
    n_features = parameters['n_features']
    n_hidden = parameters['n_hidden'] # 2
    hidden_dim = parameters['hidden_dim'] # 200
    n_class = parameters['n_class'] # 10
    learning_rate = parameters['learning_rate']

    X = tf.placeholder(tf.float32, shape=[None, n_features], name="x_input")
    Y = tf.placeholder(tf.float32, shape=[None, n_class], name="y_input")

    # Layers; Weights & Biases
    Thetas = {
        'Theta1': tf.Variable(tf.random_normal([n_features,hidden_dim])),
        'Theta2': tf.Variable(tf.random_normal([hidden_dim,hidden_dim])),
        'Theta3': tf.Variable(tf.random_normal([hidden_dim,n_class]))
        }
    Biases = {
        'Bias1': tf.Variable(tf.random_normal([1])),
        'Bias2': tf.Variable(tf.random_normal([1])),
        'Bias3': tf.Variable(tf.random_normal([1]))
        }

    # Model
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(X, Thetas['Theta1']), Biases['Bias1']))
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, Thetas['Theta2']), Biases['Bias2']))
    hypothesis = tf.add(tf.matmul(layer3, Thetas['Theta3']), Biases['Bias3'])

    # Cost & Trainer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=hypothesis))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Accuracy
    prediction = tf.round(tf.nn.softmax(hypothesis))
    correct_prediction = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Model
    model = {'X': X, 'Y': Y, 'hypothesis': hypothesis, 'cost': cost, 'train_op': train_op, 'prediction': prediction, 'accuracy': accuracy}

    return model


def neural_network(x_train, y_train, parameters, model, x_test, y_test):
    # Parameters
    training_epochs = parameters['training_epochs']

    # Cost per epoch saver
    epoch_list = []
    cost_list = []
    test_list = []

    # Save model 
    saver = tf.train.Saver()

    # Init variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Backprpagation & Cost
            _, train_cost = sess.run([model['train_op'], model['cost']], feed_dict={model['X']: x_train, model['Y']: y_train})
            test_cost = sess.run(model['cost'], feed_dict={model['X']: x_test, model['Y']: y_test})
            # Compute average loss & save in list
            #if (epoch % 10 == 0):
            epoch_list.append(epoch)
            cost_list.append(train_cost)
            test_list.append(test_cost)
            #if (epoch % 100 == 0):
            print ("Cost after epoch {0}: {1}".format(epoch, train_cost)) 
        save_path = saver.save(sess, parameters['model_path'])
        print("\nModel saved in path: {}\n".format(save_path))
    if parameters['visualize'] == True:
        plt.plot(epoch_list, cost_list)
        plt.plot(epoch_list, test_list)
        plt.show()

