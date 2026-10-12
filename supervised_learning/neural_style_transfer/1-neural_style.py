import numpy as np
import tensorflow as tf


class NST:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        # Validate style_image
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        # Validate content_image
        if not isinstance(content_image, np.ndarray) or len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        # Validate alpha
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validate beta
        if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set TensorFlow to execute eagerly
        tf.config.run_functions_eagerly(True)

        # Set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        # Validate input
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        # Calculate new dimensions to make max side 512
        if h >= w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        # Resize using bicubic interpolation
        resized = tf.image.resize(image, [new_h, new_w], method='bicubic')

        # Rescale pixel values from [0, 255] to [0, 1]
        scaled = resized / 255.0

        # Add batch dimension
        output = tf.expand_dims(scaled, axis=0)

        return output

    def load_model(self):
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg19.trainable = False

        outputs = []
        for layer_name in self.style_layers + [self.content_layer]:
            outputs.append(vgg19.get_layer(layer_name).output)

        self.model = tf.keras.Model(inputs=vgg19.input, outputs=outputs)
