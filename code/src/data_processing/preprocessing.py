from dataclasses import dataclass
import logging
from os import fspath
from pathlib import Path

import mlflow
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F

from src.tracking import configure_mlflow, log_preprocessing_state
from src.tracking.spark_mlflow_tracking import log_preprocessing_model

from src.data_processing.artifacts import PreprocessingArtifacts
from src.data_processing.feature_engineering import (assign_operating_regimes, 
                                                     drop_feature_columns, 
                                                     find_constant_feature_columns, 
                                                     fit_operating_regime_mapping)
from src.data_processing.scaling import (RegimeSensorScaler, 
                                         apply_global_sensor_scaler, 
                                         apply_sensor_scaler, 
                                         fit_regime_sensor_scaler,
                                         global_sensor_scaler_from_model)
from src.data_processing.constants import CMAPSS_SENSOR_COLUMNS, SUPPORTED_SUBSETS
from src.data_processing.split import split_train_validation_by_unit
from src.data_processing.validation import validate_cmapss_data
from src.data_processing.data_loader import add_test_rul_target, load_cmapss_test_rul, add_rul_target, load_cmapss_raw


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingRunResult:
    run_id: str
    subset: str
    train_output_path: str
    validation_output_path: str
    test_output_path: str    
    train_engine_count: int
    validation_engine_count: int
    test_engine_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    feature_count: int
    scaling_strategy: str
    regime_count: int


def input_paths(raw_data_dir, subset):
    root = Path(fspath(raw_data_dir))
    return (root / f"train_{subset}.txt",
            root / f"test_{subset}.txt",
            root / f"RUL_{subset}.txt")


def validate_input_paths(paths):
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"C-MAPSS input file not found: {path}")


def transform_with_artifacts(df, artifacts, sensor_scaler):
    assigned = assign_operating_regimes(df, artifacts.regime_mapping)
    selected = drop_feature_columns(assigned, artifacts.dropped_feature_columns)
    
    if isinstance(sensor_scaler, PipelineModel):
        return apply_global_sensor_scaler(selected, sensor_scaler, artifacts.retained_sensor_columns)
    
    return apply_sensor_scaler(selected, sensor_scaler)


def frame_counts(df):
    return df.count(), df.select("unit_id").distinct().count()


def finalize_preprocessing_output(df, sensor_columns):    
    return df.select("unit_id", "cycle", "RUL", F.col("regime").alias("operating_regime"), *sensor_columns)


def run_subset_preprocessing(spark, subset, raw_data_dir, output_dir, validation_fraction=0.2, seed=42):
    
    normalized_subset = subset.upper()
    if normalized_subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported C-MAPSS subset: {subset}")

    train_path, test_path, test_rul_path = input_paths(raw_data_dir, normalized_subset)
    validate_input_paths((train_path, test_path, test_rul_path))
    subset_output = Path(fspath(output_dir)) / normalized_subset
    
    configure_mlflow()

    with mlflow.start_run(run_name=f"{normalized_subset}-preprocessing") as run:
        LOGGER.info("starting %s", normalized_subset)
        
        raw_train = load_cmapss_raw(spark, train_path)
        raw_test = load_cmapss_raw(spark, test_path)
        
        validate_cmapss_data(raw_train)
        validate_cmapss_data(raw_test)

        full_train = add_rul_target(raw_train)
        full_test = add_test_rul_target(raw_test, load_cmapss_test_rul(spark, test_rul_path))
        
        train, validation = split_train_validation_by_unit(full_train, validation_fraction=validation_fraction, seed=seed)

        # Fit regimes before removing any constant operating setting.
        regime_mapping = fit_operating_regime_mapping(train)
        assigned_train = assign_operating_regimes(train, regime_mapping)
        
        dropped_columns = tuple(find_constant_feature_columns(assigned_train))
        
        selected_train = drop_feature_columns(assigned_train, dropped_columns).cache()
        
        sensor_columns = tuple(column for column in CMAPSS_SENSOR_COLUMNS if column in selected_train.columns)

        if normalized_subset in ["FD001", "FD003"]:
            sensor_scaler = Pipeline(stages=[
                VectorAssembler(inputCols=list(sensor_columns), outputCol="assembled_sensor_features"),
                StandardScaler(inputCol="assembled_sensor_features", outputCol="scaled_sensor_features",
                               withMean=True, withStd=True)]).fit(selected_train)
            scaling_strategy = "global"
        else:
            sensor_scaler = fit_regime_sensor_scaler(selected_train, sensor_columns)
            scaling_strategy = "regime-aware"

        artifacts = PreprocessingArtifacts(regime_mapping=regime_mapping, 
                                           dropped_feature_columns=dropped_columns, 
                                           retained_sensor_columns=sensor_columns,
                                           regime_sensor_scaler=(sensor_scaler if isinstance(sensor_scaler, RegimeSensorScaler) else None),
                                           global_sensor_scaler=(global_sensor_scaler_from_model(sensor_scaler, sensor_columns)
                                                                 if isinstance(sensor_scaler, PipelineModel) else None))

        processed_frames = []
        try:
            scaled_train = (apply_global_sensor_scaler(selected_train, sensor_scaler, artifacts.retained_sensor_columns)
                            if isinstance(sensor_scaler, PipelineModel) 
                            else apply_sensor_scaler(selected_train, sensor_scaler))
            
            scaled_validation = transform_with_artifacts(validation, artifacts, sensor_scaler)
            scaled_test = transform_with_artifacts(full_test, artifacts, sensor_scaler)
            
            processed_train = finalize_preprocessing_output(scaled_train, artifacts.retained_sensor_columns).cache()
            processed_validation = finalize_preprocessing_output(scaled_validation, artifacts.retained_sensor_columns).cache()
            processed_test = finalize_preprocessing_output(scaled_test, artifacts.retained_sensor_columns).cache()
            
            processed_frames.extend((processed_train, processed_validation, processed_test))

            train_row_count, train_engine_count = frame_counts(processed_train)
            validation_row_count, validation_engine_count = frame_counts(processed_validation)
            test_row_count, test_engine_count = frame_counts(processed_test)

            for split_name, frame in {"train": processed_train, 
                                      "validation": processed_validation, 
                                      "test": processed_test}.items():
                destination = subset_output / split_name
                frame.write.mode("overwrite").parquet(str(destination))
                LOGGER.info("wrote %s", destination)

            mlflow.log_params({"subset": normalized_subset,
                               "validation_fraction": validation_fraction,
                               "seed": seed,
                               "scaling_strategy": scaling_strategy,
                               "regime_count": len(regime_mapping)})
            
            log_preprocessing_state(artifacts)
            
            if isinstance(sensor_scaler, PipelineModel):
                log_preprocessing_model(sensor_scaler)
                
        finally:
            for frame in processed_frames:
                frame.unpersist()
                
            selected_train.unpersist()

        retained_feature_count = len(artifacts.retained_sensor_columns)
        
        result = PreprocessingRunResult(subset=normalized_subset,
                                        train_output_path=str(subset_output / "train"),
                                        validation_output_path=str(subset_output / "validation"),
                                        test_output_path=str(subset_output / "test"),
                                        run_id=run.info.run_id,
                                        train_engine_count=train_engine_count,
                                        validation_engine_count=validation_engine_count,
                                        test_engine_count=test_engine_count,
                                        train_row_count=train_row_count,
                                        validation_row_count=validation_row_count,
                                        test_row_count=test_row_count,
                                        feature_count=retained_feature_count,
                                        scaling_strategy=scaling_strategy,
                                        regime_count=len(regime_mapping))
        
        LOGGER.info("completed %s in MLflow run %s", normalized_subset, run.info.run_id)
        
        return result
