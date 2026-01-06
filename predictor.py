import numpy as np
import FinanceDataReader as fdr
import pickle

def predict_price(window_size):
    filename = "models/samsung_winsize_" + str(window_size) + ".pkl"
    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    WINDOW_SIZE = loaded['window size']
    x_scaler = loaded['x_scaler']
    y_scaler = loaded['y_scaler']
    model = loaded['model']

    # Fetch Samsung Electronics (005930) data
    df = fdr.DataReader('005930', '2022-01-01', '2026-01-07')

    df['3MA'] = np.around(df['Close'].rolling(window=3).mean(), 0)
    df['5MA'] = np.around(df['Close'].rolling(window=5).mean(), 0)
    X = df.drop(['Close', 'Volume', 'Change'], axis=1)
    X = X.dropna()
    y = df['Close']
    y = y.dropna()
    X = x_scaler.transform(X)
    y = y_scaler.transform(y.values.reshape(-1, 1))


    input_data = X[-WINDOW_SIZE:]
    input_data = input_data.reshape(1, WINDOW_SIZE, input_data.shape[1])
    predicted_price = model.predict(input_data)
    predicted_price = y_scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]