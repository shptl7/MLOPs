
# Features with string data types that will be converted to indices
CATEGORICAL_FEATURE_KEYS = ['Seasons', 'Holiday', 'Functioning_Day']

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['Temperature__C_', 'Humidity___', 'Wind_speed__m_s_', 'Visibility__10m_', 'Dew_point_temperature__C_', 'Solar_Radiation__MJ_m2_', 'Rainfall_mm_', 'Snowfall__cm_']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['Hour']

# Number of buckets used by tf.transform for encoding each bucket feature.
FEATURE_BUCKET_COUNT = {'Hour': 4}

# Feature that the model will predict
LABEL_KEY = 'Rented_Bike_Count'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
