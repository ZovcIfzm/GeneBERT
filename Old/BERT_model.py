import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import pickle
from pathlib import Path

# To do- convert this into a regression model
# To do- create a BERT model pretrained on histone modifications rather than bert_en_uncased

def build_classifier_model():
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

#input data_retrieval.retrieve_training_data() for retrieval_fn usually
def train(statistic, feature_vector, retrieval_fn, cache_path):
    # If model is saved, load model, else train model
    file_path = Path(cache_path)
    
    data = retrieval_fn(test_size = 0.25)
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_temp = np.asarray(data["X_test"])
    y_temp = np.asarray(data["y_test"])

    X_val, X_test = np.split(X_temp, int(len(files)*0.66))
    y_val, y_test = np.split(y_temp, int(len(files)*0.66))

    # Define loss function
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    # Define optimizer
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(X_train).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

    # Load BERT and train
    classifier_model = build_classifier_model() 

    classifier_model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    history = classifier_model.fit(X_train, y_train,epochs=epochs, validation_data=(X_val, y_val))

    loss, accuracy = classifier_model.evaluate(X_test, y_test)

    print("BERT classifier score: ", accuracy)
    
    # Save model
    pickle.dump(model, open(file_path, 'wb'))

    return model, score