import numpy as np
import pandas as pd
import random
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from datetime import datetime


def predict_price(ticker='005930', user_defined_features=['Open', 'High', 'Close', 'Volume', 'Change', '3MA', '5MA'], window_size=15, training_dates=('2020-01-01', datetime.now().strftime('%Y-%m-%d'))):
    """
    user_defined_features: list of features to use for training (must include 'Close')
    window_size: number of days to use for training
    training_dates: tuple of start and end dates for training
    """
    
    np.set_printoptions(precision=4, linewidth=200)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)


    df = fdr.DataReader(ticker, training_dates[0], training_dates[1])
    df['3MA'] = np.around(df['Close'].rolling(window=3).mean(), 0)
    df['5MA'] = np.around(df['Close'].rolling(window=5).mean(), 0)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', '3MA', '5MA']
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    y_train = train['Close']
    y_test = test['Close']

    features_to_drop = list(set(features) - set(user_defined_features))
    X_train = train.drop(features_to_drop, axis=1)
    X_test = test.drop(features_to_drop, axis=1)
   

    X_train = X_train.dropna()
    X_test = X_test.dropna()
    y_train = y_train[X_train.index]
    y_test = y_test[X_test.index]
    xs = MinMaxScaler()
    ys = MinMaxScaler()

    X_train_s = xs.fit_transform(X_train)
    X_test_s = xs.transform(X_test)
    y_train_s = ys.fit_transform(y_train.values.reshape(-1, 1))
    y_test_s = ys.transform(y_test.values.reshape(-1, 1))


    def make_sequential_dataset(X, y, window_size):
        X_list = []
        y_list = []
        for i in range(len(X) - window_size):
            X_list.append(X[i:i+window_size])
            y_list.append(y[i+window_size])
        return np.array(X_list), np.array(y_list)
    WINDOW_SIZE = window_size
    # hmm.. this is a very interesting feature
    X_train_seq, y_train_seq = make_sequential_dataset(X_train_s, y_train_s, WINDOW_SIZE)
    X_test_seq, y_test_seq = make_sequential_dataset(X_test_s, y_test_s, WINDOW_SIZE)
    
    model = keras.Sequential([
        keras.Input(shape=(WINDOW_SIZE, len(user_defined_features))),
        layers.LSTM(units=64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation=None)
    ])
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    EPOCHS = 30
    BATCH_SIZE = 32

    model.fit(
        X_train_seq, y_train_seq,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    input_data = df.drop(features_to_drop, axis=1)[-window_size:]
    input_data = input_data.dropna()
    input_data = xs.transform(input_data)
    input_data = input_data.reshape(1, WINDOW_SIZE, input_data.shape[1])

    
    # MC Dropout for Confidence Estimation
    n_iterations = 30
    predictions = []
    
    # We need to replicate input_data for batch prediction or loop
    # Looping is safer for memory if n is large, but for 30 it's fine.
    # Actually, we can just use training=True in a loop or call predict multiple times?
    # Keras functional API / Sequential model call with training=True works.
    
    # We need to use model(input_data, training=True) to enable dropout during inference
    # input_data is numpy array, convert to tensor if needed or just pass as is (usually works)
    input_data = tf.convert_to_tensor(input_data)
    for _ in range(n_iterations):
        # model() call expects a tensor, predict() is higher level. 
        # But predict() usually disables dropout.
        # We can use model(x, training=True)
        pred = model(input_data, training=True).numpy()
        predictions.append(pred)
        
    predictions = np.array(predictions) # Shape: (n_iterations, 1, 1)
    
    # Inverse transform all predictions
    predictions_resc = np.array([ys.inverse_transform(p) for p in predictions])
    predictions_resc = predictions_resc.reshape(n_iterations) # Shape: (n_iterations,)
    
    mean_pred = np.mean(predictions_resc)
    std_pred = np.std(predictions_resc)
    
    # Return last 20 days for plotting (unscaled)
    last_20_days = df.tail(20).copy()

    # Return mean prediction, confidence (std dev), and history
    return mean_pred, std_pred, last_20_days


if __name__ == "__main__":
    print(predict_price('066570', ['3MA', 'Volume', 'Open'], window_size=30)[0])
