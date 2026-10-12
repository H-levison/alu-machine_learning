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

        # Calculate gram matrices for style image
        self.gram_style_features = []
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        style_outputs = self.model(preprocessed_style)

        for i in range(len(self.style_layers)):
            gram = self.gram_matrix(style_outputs[i])
            self.gram_style_features.append(gram)

        # Calculate content feature for content image
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)
        content_outputs = self.model(preprocessed_content)
        self.content_feature = content_outputs[-1]

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

    @staticmethod
    def gram_matrix(input_layer):
        # Validate input
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get dimensions
        shape = tf.shape(input_layer)
        batch = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]

        # Reshape to (batch, h*w, c)
        reshaped = tf.reshape(input_layer, (batch, h*w, c))

        # Compute Gram matrix: (batch, c, c)
        gram = tf.matmul(reshaped, reshaped, transpose_a=True)

        # Normalize by h*w
        hw = tf.cast(h * w, tf.float32)
        gram = gram / hw

        return gram

    def layer_style_cost(self, style_output, gram_target):
        # Validate style_output
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        # Get number of channels
        c = style_output.shape[3]

        # Validate gram_target shape
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or gram_target.shape != (1, c, c):
            raise TypeError(f"gram_target must be a tensor of shape [1, {c}, {c}]")

        # Calculate gram matrix of style_output
        gram_style = self.gram_matrix(style_output)

        # Calculate the difference
        diff = gram_style - gram_target

        # Calculate cost: sum of squared differences / C^2
        cost = tf.reduce_sum(tf.square(diff)) / (c ** 2)

        return cost

    def style_cost(self, style_outputs):
        # Validate style_outputs
        if not isinstance(style_outputs, list) or len(style_outputs) != len(self.style_layers):
            raise TypeError(f"style_outputs must be a list with a length of {len(self.style_layers)}")

        # Calculate total style cost with equal weights
        num_layers = len(self.style_layers)
        weight = 1.0 / num_layers

        total_cost = 0
        for i, style_output in enumerate(style_outputs):
            layer_cost = self.layer_style_cost(style_output, self.gram_style_features[i])
            total_cost += weight * layer_cost

        return total_cost

    def content_cost(self, content_output):
        # Validate content_output
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or content_output.shape != self.content_feature.shape:
            raise TypeError(f"content_output must be a tensor of shape {self.content_feature.shape}")

        # Calculate the difference
        diff = content_output - self.content_feature

        # Calculate cost: mean squared error
        cost = tf.reduce_mean(tf.square(diff))

        return cost

    def total_cost(self, generated_image):
        # Validate generated_image
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError(f"generated_image must be a tensor of shape {self.content_image.shape}")

        # Preprocess the generated image
        preprocessed_generated = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)

        # Get outputs from the model
        outputs = self.model(preprocessed_generated)

        # Get style and content outputs
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        # Calculate style and content costs
        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        # Calculate total cost
        J = self.alpha * J_content + self.beta * J_style

        return (J, J_content, J_style)

    def compute_grads(self, generated_image):
        # Validate generated_image
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError(f"generated_image must be a tensor of shape {self.content_image.shape}")

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return (gradients, J_total, J_content, J_style)

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        # Validate iterations
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        # Validate step
        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and less than iterations")

        # Validate lr
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        # Validate beta1
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        # Validate beta2
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # Initialize generated image as content image
        generated_image = tf.Variable(tf.identity(self.content_image))

        # Create Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)

        # Compute initial cost
        J_total, J_content, J_style = self.total_cost(generated_image)

        # Print initial cost
        if step is not None:
            print(f"Cost at iteration 0: {J_total}, content {J_content}, style {J_style}")

        # Track best cost and image
        best_cost = J_total
        best_image = tf.identity(generated_image)

        # Perform gradient descent
        for i in range(1, iterations + 1):
            # Compute gradients
            grads, J_total, J_content, J_style = self.compute_grads(generated_image)

            # Apply gradients
            optimizer.apply_gradients([(grads, generated_image)])

            # Update best cost and image
            if J_total < best_cost:
                best_cost = J_total
                best_image = tf.identity(generated_image)

            # Print progress
            if step is not None and (i % step == 0):
                print(f"Cost at iteration {i}: {J_total}, content {J_content}, style {J_style}")

        return (best_image, best_cost)
