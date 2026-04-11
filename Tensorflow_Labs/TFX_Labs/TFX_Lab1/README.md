# TFX End-to-End Pipeline - Seoul Bike Sharing Demand

This repository implements a full Machine Learning Pipeline using **TensorFlow Extended (TFX)**. It is configured to run end-to-end on the **Seoul Bike Sharing Demand Dataset** from the UCI Machine Learning Repository, transitioning all the way from raw data ingestion into an fully evaluated Deep Neural Network (`SavedModel`).

## Pipeline Components

This pipeline covers the following TFX execution steps:
1. **ExampleGen:** Ingests the `SeoulBikeData.csv` format into standard compressed `TFRecord` protobufs and splits it into training and evaluation sets.
2. **StatisticsGen:** Generates summary statistics and profiles using *TensorFlow Data Validation (TFDV)*.
3. **SchemaGen:** Infers the structural metadata of the data, including value bounds and accepted terminology.
4. **ExampleValidator:** Runs evaluation sweeps over incoming data streams to flag any statistical anomalies or schema violations.
5. **Transform:** Operates feature engineering scaling, categorization, and feature crosses into a servable TensorFlow Graph.
6. **Trainer:** Consumes the TFRecords and Transform Graph to train a Keras Deep Neural Network. Output is a fully packed serving signature `SavedModel`.
7. **Evaluator (TFMA):** Performs granular, sliced model validation (sliced by `Seasons`) directly on the trained model.

## Advanced Feature Engineering
The pipeline boasts the following explicit mathematical injections during the Transform phase:
- **Z-Score Normalization**: Continuous meteorological features (`Temperature`, `Humidity`, `Wind_speed`, and `Solar_Radiation`) are variance-scaled using `tft.scale_to_z_score`.
- **Wind Chill Factor**: Interaction terms computed explicitly via `tf.multiply(Temperature, Wind_speed)` before passing into Z-scale normalization!
- **Date Extraction**: Uses `tf.strings.substr` natively in the graph to dynamically parse Month variables right out of the string Datetime (`DD/MM/YYYY`) format for cyclical tracking.
- **Robust Bucketization**: The continuous `Hour` (0-23) timestamp feature is grouped cleanly into chronological periods (Morning, Afternoon, Evening, Night) using `tft.bucketize`.
- **Feature Crossing**: The Transform block explicitly combinations categorical terms using `tf.strings.join`, crossing the `Hour` bin directly with `Seasons` to model intricate behavioral shifts across times of day globally per season.

## Execution
Simply execute all cells in `C2_W2_Lab_2_Feature_Engineering_Pipeline.ipynb`.
The notebook will automatically download the UCI dataset, sanitize the headers and encoding, inject `.py` training and transformer configurations directly into your disk, and run the complete DAG step-by-step.
