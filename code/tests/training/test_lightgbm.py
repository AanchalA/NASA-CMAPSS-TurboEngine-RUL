import unittest
from unittest.mock import patch

from src.training.tree_models.train_lightgbm import train_lightgbm


class LightGBMTrainingTests(unittest.TestCase):
    @patch("src.training.tree_models.train_lightgbm.train_tabular_model")
    def test_uses_fixed_parameters_and_capped_target(self, train_tabular_model_mock):
        train_tabular_model_mock.return_value = "training-run"

        run_id = train_lightgbm("FD003", "preprocessing-run", "processed")

        self.assertEqual(run_id, "training-run")
        arguments = train_tabular_model_mock.call_args.kwargs
        self.assertEqual(arguments["model_name"], "lightgbm")
        self.assertEqual(arguments["rul_cap"], 125)
        self.assertEqual(arguments["model"].n_estimators, 500)
        self.assertEqual(arguments["model"].learning_rate, 0.05)
        self.assertEqual(arguments["model"].num_leaves, 31)
        self.assertEqual(arguments["model"].subsample, 0.8)
        self.assertEqual(arguments["model"].colsample_bytree, 0.8)


if __name__ == "__main__":
    unittest.main()
