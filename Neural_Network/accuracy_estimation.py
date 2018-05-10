import tensorflow as tf
import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt

def Accuracy(parameters, model, X_train, Y_train, X_test, Y_test):
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, parameters['model_path'])
        print ("Train Accuracy: {}%".format(sess.run(model['accuracy'], feed_dict={model['X']: X_train, model['Y']: Y_train}) * 100))
        print ("Test Accuracy: {}%\n".format(sess.run(model['accuracy'], feed_dict={model['X']: X_test, model['Y']: Y_test}) * 100))

def Estimation(parameters, model, X_test):
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    i = 1
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, parameters['model_path'])
        prediction = sess.run(model['prediction'], feed_dict={model['X']: X_test})
        tmp = []
        for pred in prediction:
            nb = 0
            k = 0
            for j in pred:
                if j == 1:
                    tmp.append(nb)
                    k = 1
                nb += 1
            if k == 0:
                tmp.append(0)
    output = pd.DataFrame({'ImageId': np.arange(1, len(tmp) + 1), 'Label': tmp})
    output.to_csv("estimation.csv", index=False)
    print("Writting output to estimation.csv")

def Exemple(df_test):
    if (os.path.isfile("./estimation.scv")):
        print("File 'estimation.scv' not found.")
        return 42
    labels = pd.read_csv("./estimation.csv")
    tmp = df_test.values
    tmp = tmp.reshape(tmp.shape[0], 28, 28)
    for i in range(9):
        plt.subplot(330 + (i + 1))
        plt.imshow(tmp[i], cmap=plt.get_cmap('gray'))
        plt.title(labels['Label'][i]);
    plt.show()
