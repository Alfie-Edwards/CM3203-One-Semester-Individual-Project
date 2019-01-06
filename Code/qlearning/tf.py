from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf


# Single Agents

class Agent(ABC):
    def __init__(self, state_size, n_actions, learning_rate=0.001):
        self.state_size = state_size
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.input_states, self.predictions, self.predicted_best_actions = self.build_model()
        self.training_targets, self.training_updates, self.training_mask, self.loss = self.build_training_tensors(self.predictions)
        self.tf_session = None

    def start_session(self):
        self.tf_session = tf.Session()
        self.tf_session.run(tf.global_variables_initializer())

    def end_session(self):
        if self.tf_session is None:
            return
        self.tf_session.close()

    def train(self, states, outcomes, training_mask):
        states = self._reshape_states(states)
        feed_dict = {self.input_states: states, self.training_targets: outcomes, self.training_mask: training_mask}
        _, loss = self.tf_session.run((self.training_updates, self.loss), feed_dict=feed_dict)
        return loss

    def get_predictions(self, states):
        if not states:
            return []
        states = self._reshape_states(states)
        feed_dict = {self.input_states: states}
        return self.tf_session.run(self.predictions, feed_dict=feed_dict)

    def predict_best_actions(self, states):
        if not states:
            return []
        states = self._reshape_states(states)
        feed_dict = {self.input_states: states}
        return self.tf_session.run(self.predicted_best_actions, feed_dict=feed_dict)

    def _reshape_states(self, states):
        input_shape = [item if item is not None else -1 for item in self.input_states.get_shape().as_list()]
        states = np.reshape(states, input_shape)
        return states

    @abstractmethod
    def build_model(self):
        pass

    def build_training_tensors(self, predictions):
        training_targets = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=predictions.get_shape()))
        training_mask = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=predictions.get_shape()))
        softmax = tf.nn.softmax(training_targets)
        masked_targets = tf.multiply(softmax, training_mask)
        masked_predictions = tf.multiply(predictions, training_mask)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=masked_predictions, labels=masked_targets, dim=1)
        loss = tf.reduce_mean(cross_entropy)

        training_updates = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return training_targets, training_updates, training_mask, loss

    def save_session(self, path):
        if self.tf_session is None:
            return
        saver = tf.train.Saver()
        saver.save(self.tf_session, path)
        print("Model saved in:", path)

    def load_session(self, path):
        if self.tf_session is None:
            return
        saver = tf.train.Saver()
        saver.restore(self.tf_session, path)
        print("Model loaded from:", path)


class FeedForwardAgent(Agent):
    def __init__(self, state_size, n_actions, intermediate_layers, learning_rate=0.001):
        self.intermediate_layers = intermediate_layers
        super().__init__(state_size, n_actions, learning_rate)

    def build_model(self):
        input_states = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])

        prev_size = self.state_size
        latest_layer = input_states
        for layer in self.intermediate_layers:
            size = layer * self.state_size
            weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            latest_layer = tf.nn.tanh(tf.add(tf.matmul(latest_layer, weights), biases))
            prev_size = size

        prediction_weights = tf.Variable(
            tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.n_actions], stddev=0.1))
        prediction_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.n_actions], stddev=0.1))
        predictions = tf.add(tf.matmul(latest_layer, prediction_weights), prediction_biases)
        predicted_best_actions = tf.argmax(predictions, axis=1)

        return input_states, predictions, predicted_best_actions


class ConvolutionalAgent(Agent):
    def __init__(self, state_width, state_height, n_actions, filter_sizes, connected_sizes, learning_rate=0.001):
        self.state_width = state_width
        self.state_height = state_height
        self.filter_sizes = filter_sizes
        self.connected_sizes = connected_sizes
        super().__init__(state_width * state_height, n_actions, learning_rate)

    def build_model(self):
        input_states = tf.placeholder(dtype=tf.float32, shape=[None, self.state_width, self.state_height, 1])

        latest_layer = input_states
        n_input_channels = 1
        n_filters = 16
        for filter_size in self.filter_sizes:
            shape = [filter_size[0], filter_size[1], n_input_channels, n_filters]
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            biases = tf.Variable(tf.truncated_normal([n_filters], stddev=0.1))
            convolution = tf.nn.conv2d(latest_layer, weights, [1, 1, 1, 1], padding='SAME')
            latest_layer = tf.nn.relu(tf.nn.bias_add(convolution, biases))
            n_input_channels = n_filters
            n_filters *= 2

        prev_size = self.state_size * n_input_channels
        latest_layer = tf.reshape(latest_layer, shape=[-1, prev_size])

        for size in self.connected_sizes:
            size *= self.state_size * n_input_channels
            weights = tf.Variable(tf.truncated_normal([prev_size, size], stddev=0.1))
            biases = tf.Variable(tf.truncated_normal([size], stddev=0.1))
            latest_layer = tf.nn.relu(tf.add(tf.matmul(latest_layer, weights), biases))
            prev_size = size

        prediction_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.n_actions], stddev=0.1))
        prediction_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.n_actions], stddev=0.1))
        predictions = tf.add(tf.matmul(latest_layer, prediction_weights), prediction_biases)
        predicted_best_actions = tf.argmax(predictions, axis=1)

        return input_states, predictions, predicted_best_actions


# Double Agents

class DoubleAgent(Agent):
    def __init__(self, state_size, n_actions, comms_size, learning_rate=0.001):
        self.comms_size = comms_size
        self.comms_1_to_2 = None
        self.comms_2_to_1 = None
        super().__init__(state_size, n_actions, learning_rate)

    def predict_best_actions(self, states):
        if not states:
            return [], [], []
        states = self._reshape_states(states)
        feed_dict = {self.input_states: states}
        return self.tf_session.run([self.predicted_best_actions, self.comms_1_to_2, self.comms_2_to_1], feed_dict=feed_dict)

    @abstractmethod
    def build_model(self):
        pass

    def build_training_tensors(self, predictions):
        training_targets = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=predictions.get_shape()))
        training_mask = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=predictions.get_shape()))
        softmax = tf.nn.softmax(training_targets)
        masked_targets = tf.multiply(softmax, training_mask)
        masked_predictions = tf.multiply(predictions, training_mask)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=masked_predictions, labels=masked_targets, dim=2)
        loss = tf.reduce_mean(cross_entropy)

        training_updates = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return training_targets, training_updates, training_mask, loss


class DoubleFeedForwardAgent(DoubleAgent):
    def __init__(self, state_size, n_actions, comms_size, pre_comms_layers, comms_layers, post_comms_layers, learning_rate=0.001):
        self.pre_comms_layers = pre_comms_layers
        self.comms_layers = comms_layers
        self.post_comms_layers = post_comms_layers
        super().__init__(state_size, n_actions, comms_size, learning_rate)

    def build_model(self):
        combined_input_states = tf.placeholder(dtype=tf.float32, shape=[None, 2, self.state_size])
        input_state_1, input_state_2 = tf.split(combined_input_states, num_or_size_splits=2, axis=1)

        latest_layer_1 = tf.reshape(input_state_1, shape=[-1, self.state_size])
        latest_layer_2 = tf.reshape(input_state_2, shape=[-1, self.state_size])
        prev_size = self.state_size

        for layer in self.pre_comms_layers:
            size = layer * self.state_size
            weights_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            weights_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            biases_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            biases_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            latest_layer_1 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_1, weights_1), biases_1))
            latest_layer_2 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_2, weights_2), biases_2))
            prev_size = size

        pre_comms_1 = latest_layer_1
        pre_comms_2 = latest_layer_2

        for layer in self.comms_layers:
            size = layer * self.state_size
            weights_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            weights_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            biases_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            biases_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            latest_layer_1 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_1, weights_1), biases_1))
            latest_layer_2 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_2, weights_2), biases_2))
            prev_size = size

        comms_1_to_2_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.comms_size], stddev=0.1))
        comms_2_to_1_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.comms_size], stddev=0.1))
        comms_1_to_2_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.comms_size], stddev=0.1))
        comms_2_to_1_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.comms_size], stddev=0.1))
        self.comms_1_to_2 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_1, comms_1_to_2_weights), comms_1_to_2_biases))
        self.comms_2_to_1 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_2, comms_2_to_1_weights), comms_2_to_1_biases))
        latest_layer_1 = tf.concat([pre_comms_1, self.comms_2_to_1], axis=1)
        latest_layer_2 = tf.concat([pre_comms_2, self.comms_1_to_2], axis=1)
        prev_size = self.pre_comms_layers[-1] * self.state_size + self.comms_size

        for layer in self.post_comms_layers:
            size = layer * self.state_size
            weights_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            weights_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            biases_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            biases_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            latest_layer_1 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_1, weights_1), biases_1))
            latest_layer_2 = tf.nn.tanh(tf.add(tf.matmul(latest_layer_2, weights_2), biases_2))
            prev_size = size

        prediction_weights_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.n_actions], stddev=0.1))
        prediction_weights_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.n_actions], stddev=0.1))
        prediction_biases_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.n_actions], stddev=0.1))
        prediction_biases_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.n_actions], stddev=0.1))
        predictions_1 = tf.add(tf.matmul(latest_layer_1, prediction_weights_1), prediction_biases_1)
        predictions_2 = tf.add(tf.matmul(latest_layer_2, prediction_weights_2), prediction_biases_2)
        predictions_1 = tf.reshape(predictions_1, shape=[-1, 1, self.n_actions])
        predictions_2 = tf.reshape(predictions_2, shape=[-1, 1, self.n_actions])

        combined_predictions = tf.concat([predictions_1, predictions_2], axis=1)
        combined_predicted_best_actions = tf.argmax(combined_predictions, axis=2)

        return combined_input_states, combined_predictions, combined_predicted_best_actions


class DoubleConvolutionalAgent(DoubleAgent):
    def __init__(self, state_width, state_height, n_actions, comms_size, filter_sizes, comms_sizes, connected_sizes, learning_rate=0.001):
        self.state_width = state_width
        self.state_height = state_height
        self.filter_sizes = filter_sizes
        self.comms_sizes = comms_sizes
        self.connected_sizes = connected_sizes
        super().__init__(state_width * state_height, n_actions, comms_size, learning_rate)

    def build_model(self):
        combined_input_states = tf.placeholder(dtype=tf.float32, shape=[None, 2, self.state_width, self.state_height, 1])
        input_state_1, input_state_2 = tf.split(combined_input_states, num_or_size_splits=2, axis=1)

        latest_layer_1 = tf.reshape(input_state_1, shape=[-1, self.state_width, self.state_height, 1])
        latest_layer_2 = tf.reshape(input_state_2, shape=[-1, self.state_width, self.state_height, 1])
        n_input_channels = 1
        n_filters = 16
        for filter_size in self.filter_sizes:
            shape = [filter_size[0], filter_size[1], n_input_channels, n_filters]
            weights_1 = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            weights_2 = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            biases_1 = tf.Variable(tf.truncated_normal([n_filters], stddev=0.1))
            biases_2 = tf.Variable(tf.truncated_normal([n_filters], stddev=0.1))
            convolution_1 = tf.nn.conv2d(latest_layer_1, weights_1, [1, 1, 1, 1], padding='SAME')
            convolution_2 = tf.nn.conv2d(latest_layer_2, weights_2, [1, 1, 1, 1], padding='SAME')
            latest_layer_1 = tf.nn.relu(tf.nn.bias_add(convolution_1, biases_1))
            latest_layer_2 = tf.nn.relu(tf.nn.bias_add(convolution_2, biases_2))
            n_input_channels = n_filters
            n_filters *= 2

        flattened_size = self.state_size * n_input_channels
        latest_layer_1 = tf.reshape(latest_layer_1, shape=[-1, flattened_size])
        latest_layer_2 = tf.reshape(latest_layer_2, shape=[-1, flattened_size])
        pre_comms_1 = latest_layer_1
        pre_comms_2 = latest_layer_2

        prev_size = flattened_size

        for size in self.comms_sizes:
            size *= flattened_size
            comms_1_to_2_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            comms_2_to_1_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, size], stddev=0.1))
            comms_1_to_2_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            comms_2_to_1_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[size], stddev=0.1))
            latest_layer_1 = tf.nn.relu(tf.add(tf.matmul(latest_layer_1, comms_1_to_2_weights), comms_1_to_2_biases))
            latest_layer_2 = tf.nn.relu(tf.add(tf.matmul(latest_layer_2, comms_2_to_1_weights), comms_2_to_1_biases))
            prev_size = size

        comms_1_to_2_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.comms_size], stddev=0.1))
        comms_2_to_1_weights = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.comms_size], stddev=0.1))
        comms_1_to_2_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.comms_size], stddev=0.1))
        comms_2_to_1_biases = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.comms_size], stddev=0.1))
        self.comms_1_to_2 = tf.add(tf.matmul(latest_layer_1, comms_1_to_2_weights), comms_1_to_2_biases)
        self.comms_2_to_1 = tf.add(tf.matmul(latest_layer_2, comms_2_to_1_weights), comms_2_to_1_biases)

        latest_layer_1 = tf.concat([pre_comms_1, self.comms_2_to_1], axis=1)
        latest_layer_2 = tf.concat([pre_comms_2, self.comms_1_to_2], axis=1)
        prev_size = flattened_size + self.comms_size

        for size in self.connected_sizes:
            size *= (flattened_size + self.comms_size)
            weights_1 = tf.Variable(tf.truncated_normal([prev_size, size], stddev=0.1))
            weights_2 = tf.Variable(tf.truncated_normal([prev_size, size], stddev=0.1))
            biases_1 = tf.Variable(tf.truncated_normal([size], stddev=0.1))
            biases_2 = tf.Variable(tf.truncated_normal([size], stddev=0.1))
            latest_layer_1 = tf.nn.relu(tf.add(tf.matmul(latest_layer_1, weights_1), biases_1))
            latest_layer_2 = tf.nn.relu(tf.add(tf.matmul(latest_layer_2, weights_2), biases_2))
            prev_size = size

        prediction_weights_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.n_actions], stddev=0.1))
        prediction_weights_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[prev_size, self.n_actions], stddev=0.1))
        prediction_biases_1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.n_actions], stddev=0.1))
        prediction_biases_2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[self.n_actions], stddev=0.1))
        predictions_1 = tf.add(tf.matmul(latest_layer_1, prediction_weights_1), prediction_biases_1)
        predictions_2 = tf.add(tf.matmul(latest_layer_2, prediction_weights_2), prediction_biases_2)
        predictions_1 = tf.reshape(predictions_1, shape=[-1, 1, self.n_actions])
        predictions_2 = tf.reshape(predictions_2, shape=[-1, 1, self.n_actions])

        combined_predictions = tf.concat([predictions_1, predictions_2], axis=1)
        combined_predicted_best_actions = tf.argmax(combined_predictions, axis=2)

        return combined_input_states, combined_predictions, combined_predicted_best_actions