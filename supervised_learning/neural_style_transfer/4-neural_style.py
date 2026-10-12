#!/usr/bin/env python3
"""Defines the NST class that performs neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer.

    Public class attributes:
        style_layers: the VGG19 layers used for the style extraction.
        content_layer: the VGG19 layer used for the content extraction.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes a neural style transfer instance.

        Args:
            style_image: numpy.ndarray of shape (h, w, 3) containing
                the image used as a style reference.
            content_image: numpy.ndarray of shape (h, w, 3) containing
                the image used as a content reference.
            alpha: the weight for the content cost.
            beta: the weight for the style cost.

        Raises:
            TypeError: if an image is not a numpy.ndarray of shape
                (h, w, 3), or if alpha or beta is not a non-negative
                number.
        """
        if not isinstance(style_image, np.ndarray) or \
                style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
                content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image so that its largest side is 512 pixels.

        The image is resized with bicubic interpolation and its pixel
        values are rescaled from the range [0, 255] to [0, 1].

        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the
                image to be scaled.

        Returns:
            A tf.Tensor of shape (1, h_new, w_new, 3) containing the
            scaled image.

        Raises:
            TypeError: if image is not a numpy.ndarray of shape
                (h, w, 3).
        """
        if not isinstance(image, np.ndarray) or \
                image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[0], image.shape[1]
        if h > w:
            h_new, w_new = 512, int(w * 512 / h)
        else:
            h_new, w_new = int(h * 512 / w), 512

        scaled = tf.expand_dims(image, axis=0)
        scaled = tf.image.resize_bicubic(scaled, (h_new, w_new))
        scaled = scaled / 255
        return tf.clip_by_value(scaled, 0, 1)

    def load_model(self):
        """Creates the model used to calculate the cost.

        The model uses VGG19 as a base, with its max pooling layers
        replaced by average pooling layers. Its outputs are the
        outputs of the style layers followed by the content layer.
        The model is saved in the instance attribute model.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.save("VGG19_base_model")
        custom = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model("VGG19_base_model",
                                         custom_objects=custom)
        for layer in vgg.layers:
            layer.trainable = False

        outputs = [vgg.get_layer(name).output
                   for name in self.style_layers + [self.content_layer]]
        self.model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer output.

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)
                containing the layer output.

        Returns:
            A tf.Tensor of shape (1, c, c) containing the gram matrix
            of input_layer.

        Raises:
            TypeError: if input_layer is not a tensor of rank 4.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        shape = tf.shape(input_layer)
        features = tf.reshape(input_layer, (shape[0], -1, shape[3]))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(shape[1] * shape[2], tf.float32)

    def generate_features(self):
        """Extracts the features used to calculate the cost.

        Sets the instance attributes gram_style_features, the list of
        gram matrices of the style layer outputs of the style image,
        and content_feature, the content layer output of the content
        image.
        """
        preprocess = tf.keras.applications.vgg19.preprocess_input
        style_outputs = self.model(preprocess(self.style_image * 255))
        content_outputs = self.model(preprocess(self.content_image * 255))
        self.gram_style_features = [self.gram_matrix(output)
                                    for output in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer.

        Args:
            style_output: tf.Tensor of shape (1, h, w, c) containing
                the layer style output of the generated image.
            gram_target: tf.Tensor of shape (1, c, c) containing the
                gram matrix of the target style output for that layer.

        Returns:
            The style cost of the layer.

        Raises:
            TypeError: if style_output is not a tensor of rank 4 or
                gram_target is not a tensor of shape (1, c, c).
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = int(style_output.shape[-1])
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape.as_list() != [1, c, c]:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))
