"""Airflow-runtime tests; run only in the existing WSL Airflow environment."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import pytest

from src.dags.airflow_config import PROJECT_ROOT


DAG_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "dags"
    / "cmapss_preprocessing.py"
)
SPEC = importlib.util.spec_from_file_location("cmapss_preprocessing_dag", DAG_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CmapssPreprocessingDagTests(unittest.TestCase):
    def test_dag_id_tasks_and_orchestration_graph(self) -> None:
        dag = MODULE.cmapss_preprocessing
        self.assertEqual(dag.dag_id, "cmapss_preprocessing")
        self.assertIsNone(dag.schedule)
        self.assertEqual(
            set(dag.task_ids),
            {
                "resolve_runtime",
                "select_subsets",
                "preprocess_subset",
                "collect_run_ids",
            },
        )
        self.assertEqual(dag.params["subset"], "FD001")
        self.assertEqual(
            dag.get_task("resolve_runtime").downstream_task_ids,
            {"select_subsets", "preprocess_subset"},
        )
        self.assertEqual(
            dag.get_task("select_subsets").downstream_task_ids,
            {"preprocess_subset", "collect_run_ids"},
        )
        preprocessing_task = dag.get_task("preprocess_subset")
        self.assertEqual(
            preprocessing_task.upstream_task_ids,
            {"resolve_runtime", "select_subsets"},
        )
        self.assertEqual(
            preprocessing_task.downstream_task_ids,
            {"collect_run_ids"},
        )
        self.assertEqual(
            dag.get_task("collect_run_ids").upstream_task_ids,
            {"select_subsets", "preprocess_subset"},
        )

    @pytest.mark.requires_data
    def test_runtime_uses_native_data_directory_defaults(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CMAPSS_RAW_DATA_DIR": str(
                    PROJECT_ROOT / "Data" / "CMAPSSData"
                ),
                "CMAPSS_PROCESSED_DATA_DIR": str(
                    PROJECT_ROOT / "Data" / "processed"
                ),
            },
        ):
            runtime = MODULE.cmapss_preprocessing.get_task(
                "resolve_runtime"
            ).python_callable()

        self.assertEqual(
            runtime["raw_data_dir"],
            str(PROJECT_ROOT / "Data" / "CMAPSSData"),
        )
        self.assertEqual(
            runtime["processed_data_dir"],
            str(PROJECT_ROOT / "Data" / "processed"),
        )


if __name__ == "__main__":
    unittest.main()
