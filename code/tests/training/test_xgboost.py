import unittest
from unittest.mock import patch

from src.training.tree_models.train_xgboost import train_xgboost


class XGBoostTrainingTests(unittest.TestCase):
    @patch("src.training.tree_models.train_xgboost.train_tabular_model")
    def test_uses_fixed_parameters(self, train_tabular_model_mock):
        train_tabular_model_mock.return_value = "training-run"

        run_id = train_xgboost("FD003", "preprocessing-run", "processed")

        self.assertEqual(run_id, "training-run")
        arguments = train_tabular_model_mock.call_args.kwargs
        self.assertEqual(arguments["model_name"], "xgboost")
        self.assertNotIn("rul_cap", arguments)
        self.assertEqual(arguments["model"].n_estimators, 500)
        self.assertEqual(arguments["model"].max_depth, 6)
        self.assertEqual(arguments["model"].learning_rate, 0.05)
        self.assertEqual(arguments["model"].subsample, 0.8)
        self.assertEqual(arguments["model"].colsample_bytree, 0.8)


if __name__ == "__main__":
    unittest.main()
