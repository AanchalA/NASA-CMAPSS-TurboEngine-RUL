import unittest

import numpy as np
import torch

from src.training import LSTMRegressor
from src.training.lstm import predict_sequences


class LSTMTests(unittest.TestCase):
    def test_predicts_one_life_ratio_per_sequence(self):
        model = LSTMRegressor(feature_count=3)
        sequences = np.zeros((4, 30, 3), dtype=np.float32)

        predictions = predict_sequences(model, sequences, batch_size=2)

        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(np.isfinite(predictions).all())

    def test_model_has_the_requested_single_lstm_layer(self):
        model = LSTMRegressor(feature_count=15)

        self.assertEqual(model.lstm.hidden_size, 64)
        self.assertEqual(model.lstm.num_layers, 1)
        self.assertEqual(model.lstm.dropout, 0)
        self.assertIsInstance(model.output, torch.nn.Linear)

    def test_predicts_variable_length_sequences_in_input_order(self):
        model = LSTMRegressor(feature_count=3)
        sequences = [
            np.zeros((30, 3), dtype=np.float32),
            np.zeros((23, 3), dtype=np.float32),
            np.zeros((30, 3), dtype=np.float32),
        ]

        predictions = predict_sequences(model, sequences, batch_size=2)

        self.assertEqual(predictions.shape, (3,))
        self.assertTrue(np.isfinite(predictions).all())

if __name__ == "__main__":
    unittest.main()
