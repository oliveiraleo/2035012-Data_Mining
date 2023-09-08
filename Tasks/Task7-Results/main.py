from glob import glob
import pandas as pd
import datetime
import time
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

import signal
import multiprocessing

import os
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['GOTO_NUM_THREADS'] = '10'
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['openmp'] = 'True'

warnings.filterwarnings("ignore") # Disables warnings

###############
# Global Vars #
###############
# data = [] # creates an empty list
tf.random.set_seed(7)

##############################
# Useful auxiliary functions #
##############################
def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def join_data_files():
    data = [] # creates an empty list
    print("[INFO] Loading the input data... ", end='')
    files = glob("./Tasks/Task7-Results/Shanghai-Telcome-Six-Months-DataSet/*.xlsx")
    files.sort()

    for f in files[:]:
        df = pd.read_excel(f, decimal=",")
        df = df.dropna() # removes all the instances with empty data
        data.append(df)
    
    data = pd.concat(data, ignore_index=True)
    data.to_csv("./Tasks/Task7-Results/results/task7-data-complete.csv", index=False)

    return data

def load_data():
    data = [] # creates an empty list
    print("[INFO] Loading the input data... ", end='')
    
    # data = join_data_files() # Joins the data of the data set (executing this is only required once)

    try:
        data = pd.read_csv("./Tasks/Task7-Results/results/task7-data-complete.csv") # reads the full data set data
    except FileNotFoundError:
        print("[ FAIL ]")
        print("[ERROR] Make sure to join the data first")

    # print(data) # DEBUG
    print("[ OK ]")
    return data

def preprocessing(data):
    # splits the 'month' feature to get the year then removes it from the feature
    data["month"] = data["month"].astype(str)
    data["year"] = data["month"].str[:4].astype(int)
    data["month"] = data["month"].str[-2:].astype(int)
    # reorders the features on the df
    data = data[["year", "month", "date", "latitude", "longitude", "user id"]]

    # remove the duplicated instances
    # print("Before: ", len(data)) # DEBUG
    data.drop_duplicates(inplace=True, ignore_index=True)
    # print("After:", len(data)) # DEBUG

    # counts the number of users per day on each antenna
    data_agg = df.groupby(["year", "month", "date", "latitude", "longitude"]).size().reset_index(name='count')

    # print(data_agg["count"].head(n=10)) # DEBUG

    # changes the index to a timestamp instead of a int
    data_agg = data_agg.rename(columns={"date":"day"}) # renames the column to use the pd.to_datetime function
    data_agg["Timestamp"] = pd.to_datetime(data_agg[["year", "month", "day"]])
    data_agg = data_agg.drop(["year", "month", "day"], axis=1) # removes the old feature data
    data_agg = data_agg[["data", "latitude", "longitude", "count"]]
    # print(data_agg.head(n=10)) # DEBUG

    return data_agg

def plot_number_users_for_each_antenna(data):
    for _, df in data.groupby(["latitude", "longitude"]):
        df = df.reset_index()
        print(df.head(n=30))

        # print(data[["year", "month", "date"]])

        # df = df.rename(columns={"date":"day"})
        
        # df["time"] = pd.to_datetime(df[["year", "month", "day"]])

        plt.plot(df["time"], df["count"], "o-")
        plt.xticks(rotation=45)
        plt.show()

def time_series_prediction(data):
    # print("Data:", data)

    # split into train and test sets
    train_size = int(len(data) * 0.7) # 70% train
    test_size = len(data) - train_size # 30% test

    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # X_train, X_test, y_train, y_test = train_test_split(data[["latitude", "longitude"]], data["count"], test_size=0.3, shuffle=False) 
    # def create_sequences(data, seq_length):
    #     sequences = []
    #     for i in range(len(data) - seq_length):
    #         seq = data[i:i + seq_length]
    #         sequences.append(seq)
    #     return np.array(sequences) 
    # X_train = create_sequences(X_train, 10)
    # X_test = create_sequences(X_test, 10)

    X_train = train[:,:2]
    X_test = test[:,:2]

    # print("X_train: ", X_train)
    # X_train = X_train.reshape((len(X_train), 1, 2))
    # X_test = X_test.reshape((len(X_test), 1, 2))
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # y_train = train[:,2]
    # y_test = test[:,2]
    y_train = train[:,0]
    y_test = test[:,0]


    model = Sequential()

    batch_size=2**4
    epochs = 20
    patience = 50
    
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)

    # TODO tentar paralelizar o LSTM tbm
    # model.add(LSTM(4, input_shape=(len(X_train), 2)))
    model.add(LSTM(16, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    # model.add(LSTM(4, input_shape=(None, 2)))
    # model.add(LSTM(4, input_shape=(None, 2)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
    fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es], validation_data=(X_test, y_test))
    # fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))
    # score = model.evaluate(X_test, y_test)
    pred = model.predict(X_test)

    score = r2_score(y_test, pred)
    print('R2:', round(score,4))

    print(score)

    # plt.plot(fit.history['loss'], label='treino')
    # plt.plot(fit.history['val_loss'], label='teste')
    # plt.legend()
    # plt.xlabel('Epocas')
    # plt.ylabel("MSE")
    # plt.savefig('fit.png',bbox_inches='tight',dpi=500)
    # plt.savefig('fit.svg',bbox_inches='tight')
    
    # plt.plot(y_test, y_test, 'r-')
    # plt.plot(y_test, pred, 'ko')
    # plt.xlabel('Epocas')
    # plt.ylabel("MSE")
    
    plt.show()
    
    return pred


def main():
    num_workers = 1

    data = load_data()
    # data = preprocessing(data)
    # plot_number_users_for_each_antenna(data)
    # time_series_prediction(data)

    for _, df in data.groupby(["latitude", "longitude"]):
        df = df.drop(["latitude", "longitude"], axis=1)
        df = preprocessing(df)
        # print("Head:", df)
        time_series_prediction(df)

    # pool = multiprocessing.Pool(num_workers, init_worker)
    # pool.map(time_series_prediction, data)

# Calls the main functionality
if __name__ == "__main__":
    main()