#!/usr/bin/env python3
"""Evaluates the output of a saved neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Loads a saved model and returns its prediction, accuracy, and loss"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        feed = {x: X, y: Y}
        y_pred_out, accuracy_out, loss_out = sess.run(
            [y_pred, accuracy, loss], feed_dict=feed)

    return y_pred_out, accuracy_out, loss_out
