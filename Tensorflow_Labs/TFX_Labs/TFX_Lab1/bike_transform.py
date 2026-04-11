
import tensorflow as tf
import tensorflow_transform as tft

import bike_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = bike_constants.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = bike_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = bike_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = bike_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = bike_constants.LABEL_KEY
_transformed_name = bike_constants.transformed_name

# Define the transformations
def preprocessing_fn(inputs):
    outputs = {}

    month_str = tf.strings.substr(inputs['Date'], 3, 2)
    outputs['Month_xf'] = tft.compute_and_apply_vocabulary(month_str)

    # Temperature * Wind_speed
    wind_chill_raw = tf.multiply(inputs['Temperature__C_'], inputs['Wind_speed__m_s_'])
    outputs['Wind_Chill_xf'] = tft.scale_to_z_score(wind_chill_raw)

    # Scale these features using z-score
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
    
    # Bucketize these features
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], _FEATURE_BUCKET_COUNT[key])

    # Convert strings to indices in a vocabulary
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Extract the string representation of Bucketized Hour
    hour_bucket_name = _transformed_name('Hour')
    hour_str = tf.strings.as_string(outputs[hour_bucket_name])

    # Cross exactly with Seasons input
    season_str = inputs['Seasons']
    crossed_feature = tf.strings.join([hour_str, season_str], separator='_')
    
    # Apply vocabulary to the crossed feature
    outputs['Hour_cross_Seasons_xf'] = tft.compute_and_apply_vocabulary(crossed_feature)

    # Cast label to float
    outputs[_transformed_name(_LABEL_KEY)] = tf.cast(inputs[_LABEL_KEY], tf.float32)

    return outputs
