import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import bike_constants

_LABEL_KEY = bike_constants.LABEL_KEY
_transformed_name = bike_constants.transformed_name

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=_gzip_reader_fn,
        label_key=_transformed_name(_LABEL_KEY))
    return dataset


def _build_keras_model(tf_transform_output, hidden_units=[64, 32]):
    feature_spec = tf_transform_output.transformed_feature_spec()
    feature_spec.pop(_transformed_name(_LABEL_KEY))
    
    inputs = {}
    for key, spec in feature_spec.items():
        inputs[key] = tf.keras.layers.Input(shape=spec.shape, name=key, dtype=spec.dtype)

    flattened_inputs = []
    for key, tensor in inputs.items():
        if tensor.dtype != tf.float32:
            flattened_inputs.append(tf.cast(tensor, tf.float32))
        else:
            flattened_inputs.append(tensor)
            
    concat = tf.keras.layers.Concatenate()(flattened_inputs)
    x = concat
    for layer_size in hidden_units:
        x = tf.keras.layers.Dense(layer_size, activation='relu')(x)
    
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files[0], tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files[0], tf_transform_output)

    model = _build_keras_model(tf_transform_output)
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=1) # Just 1 epoch for PoC fast execution

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
