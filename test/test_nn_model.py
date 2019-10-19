import unittest

from nn_model import plot_model_with_little_data


class TestNNModel(unittest.TestCase):
    def test_plot_model_with_little_data(self):
        plot_model_with_little_data()


if __name__ == "__main__":
    unittest.main()
