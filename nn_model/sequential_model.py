import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.utils import plot_model

from settings import DIR_weight, DIR_model_plot


class Sequential(Sequential):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def add_conv_act_maxpool_2x2(self, conv_filter_num, conv_core_size, act_type, maxpool_core_size):
        self.add(Conv2D(conv_filter_num, conv_core_size))
        self.add(Activation(act_type))
        self.add(MaxPooling2D(pool_size=maxpool_core_size))

    def save_weights(self, weight_file_name):
        self.save_weights(os.path.join(DIR_weight, weight_file_name))

    def plot(self):
        plot_path = os.path.join(DIR_model_plot, f'{self.model_name}.png')
        plot_model(self, plot_path, show_shapes=True)
