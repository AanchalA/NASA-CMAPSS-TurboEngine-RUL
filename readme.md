# Predictive Maintenance System for Turbofan Engines using Remaining Useful Life Estimation

Predicting the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor data.

Model Input: Last N cycles of sensor measurements for an engine
Model Output: Estimated RUL - the predicted number of operating cycles before failure.

## DATASET

- NASA PCoE Data Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

NASA provides the original CMAPSSData.zip. The dataset comes from the **NASA Ames Prognostics Center of Excellence (PCoE)** and contains **four simulated turbofan degradation datasets** under different operating conditions and fault modes.

NASA's C-MAPSS Turbofan Engine Degradation dataset contains **run-to-failure time-series data from simulated aircraft engines**. Each engine produces operational settings and sensor readings over repeated cycles. 
- Training engines are observed until failure; 
- test engines stop before failure, 
- and the task is to predict how many operational cycles remain.
The raw data has multiple engine trajectories, operational conditions, sensor noise, and degradation over time. NASA provides four subsets with different combinations of operating conditions and fault modes.
