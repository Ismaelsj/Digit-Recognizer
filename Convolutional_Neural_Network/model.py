import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def random_batches(X, Y, batches_size):
    X = X.values
    batches = []
    m, _ = X.shape
        # Shuffle training set
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape([m, 10])
    nb_batches = int(m / batches_size)
    for i in range(0 ,nb_batches):
        batche_x = shuffled_X[i * batches_size : (i + 1) * batches_size]
        batche_y = shuffled_Y[i * batches_size : (i + 1) * batches_size]
        batche = (batche_x, batche_y)
        batches.append(batche)

    if m % nb_batches != 0:
        batche_x = shuffled_X[nb_batches * batches_size :]
        batche_y = shuffled_Y[nb_batches * batches_size :]
        batche = (batche_x, batche_y)
        batches.append(batche)
        nb_batches += 1

    return batches, nb_batches

def make_model(parameters):
        # Network parmaeters
    n_features = parameters['n_features']
    kernel_size = parameters['kernel_size']
    n_channels = parameters['n_channels']
    padding = parameters['padding']
    stride = parameters['stride']
    n_filter = parameters['n_filters']
    n_hidden = parameters['n_hidden']
    hidden_dim = parameters['hidden_dim']
    n_class = parameters['n_class']
    learning_rate = parameters['learning_rate']

    X = tf.placeholder(tf.float32, [None, n_features])
    Y = tf.placeholder(tf.float32, [None, n_class])

        # Layers; Weights & Biases
    Weights = {
        'w_conv1': tf.Variable(tf.random_normal([5, 5, n_channels, 32])),
        'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'w_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        'w_output': tf.Variable(tf.random_normal([1024, n_class]))
        }
    Biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'b_output': tf.Variable(tf.random_normal([n_class]))
        }

    x = tf.reshape(X, shape=[-1, 28, 28, 1])

        # Activation
    conv1 = tf.nn.conv2d(x, Weights['w_conv1'], strides=[1, 1, 1, 1], padding=padding)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

    conv2 = tf.nn.conv2d(pool1, Weights['w_conv2'], strides=[1, 1, 1, 1], padding=padding)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

    fc = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc ,Weights['w_fc']) + Biases['b_fc'])

    hypothesis = tf.add(tf.matmul(fc, Weights['w_output']), Biases['b_output'])

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
            batches_train, nb_batches_train = random_batches(x_train, y_train, parameters['batches_size'])
            #batches_test, nb_batches_test = random_batches(x_test, y_test, parameters['batches_size'])
            for batche in batches_train:
                (batche_x, batche_y) = batche
                _, train_cost = sess.run([model['train_op'], model['cost']], feed_dict={model['X']: batche_x, model['Y']: batche_y})
                #test_cost = sess.run(model['cost'], feed_dict={model['X']: x_test, model['Y']: y_test})
                # Compute average loss & save in list
                epoch_list.append(epoch)
                cost_list.append(train_cost)
                #test_list.append(test_cost)
            print ("Cost after epoch {0}: {1}".format(epoch, train_cost))
        save_path = saver.save(sess, parameters['model_path'])
        print("\nModel saved in path: {}\n".format(save_path))

    if parameters['visualize'] == True:
        plt.plot(epoch_list, cost_list)
        #plt.plot(epoch_list, test_list)
        plt.show()

