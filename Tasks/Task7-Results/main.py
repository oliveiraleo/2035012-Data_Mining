from glob import glob
import pandas as pd
import numpy as np
import datetime
import time
import csv
import warnings
import matplotlib.pyplot as plt

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from math import sqrt

warnings.filterwarnings("ignore") # Disables warnings

###############
# Global Vars #
###############

##############################
# Useful auxiliary functions #
##############################
def join_data_files():
    data = [] # creates an empty list
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
        print("[ OK ]")
    except FileNotFoundError:
        print("[ FAIL ]")
        print("[ERROR] Make sure to join the data first")

    # print(data) # DEBUG
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
    data_agg = data.groupby(["year", "month", "date", "latitude", "longitude"]).size().reset_index(name='count')
    # print(data_agg["count"].head(n=10)) # DEBUG

    # creates a timestamp as a feature then drop the int type ones
    data_agg = data_agg.rename(columns={"date":"day"}) # renames the column to use the pd.to_datetime function
    data_agg["timestamp"] = pd.to_datetime(data_agg[["year", "month", "day"]])
    data_agg = data_agg.drop(["year", "month", "day"], axis=1) # removes the old feature data
    data_agg = data_agg[["timestamp", "latitude", "longitude", "count"]]
    # print(data_agg.head(n=10)) # DEBUG

    return data_agg

def plot_number_users_for_each_antenna(data): # TODO check this function here
    for _, df in data.groupby(["latitude", "longitude"]):
        df = df.reset_index()
        print(df.head(n=30))

        # print(data[["year", "month", "date"]])

        # df = df.rename(columns={"date":"day"})
        
        # df["time"] = pd.to_datetime(df[["year", "month", "day"]])

        plt.plot(df["time"], df["count"], "o-")
        plt.xticks(rotation=45)
        plt.show()

def time_series_prediction(df, antenna_id):
    def difference(dataset):
         diff = list()
         for i in range(1, len(dataset)):
             value = dataset[i] - dataset[i - 1]
             diff.append(value)
         return np.array(diff)
     
     
    def predict(coef, history):
         yhat = coef[0]
         for i in range(1, len(coef)):
             yhat += coef[i] * history[-i]
         return yhat

    # changes the index to a timestamp instead of a int
    df.set_index('timestamp', inplace=True)
    # normalize the dataset
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # print("Df preview:\n", df.head(20)) # DEBUG
    X = difference(df.values) # apply a differentiation on the values
    # X = df.values
    size = int(len(X) * 0.7) # calculates the train size using 70% of the data set
    train, test = X[0:size], X[size:] # splits the data set in train and test data

    # automatically determine the 'lags' parameter (this was used to discover lags values below)
    # for more info visit: https://www.statsmodels.org/devel/generated/statsmodels.tsa.ar_model.ar_select_order.html?highlight=ar_select#statsmodels-tsa-ar-model-ar-select-order
    # lags_model = ar_select_order(train, maxlag=20, glob=True)
    # lags = lags_model.ar_lags
    lags = [6]

    # trains autoregression model / for more info visit: https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html
    model = AutoReg(train, lags=lags) 
    model_fit = model.fit()
    coef = model_fit.params
    # walk forward over time steps in test (i.e. predicts the data series for the train's data amount of time)
    history = [train[i] for i in range(len(train))]
    predictions = list()
    
    for t in range(len(test)):
         yhat = predict(coef, history)
         obs = test[t]
         predictions.append(yhat)
         history.append(obs)

    # calculates the statistical evaluation variables
    rmse = sqrt(mean_squared_error(test, predictions))
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    metrics = [rmse, r2, mae]
    print("\n[INFO] Antenna %i stats" % antenna_id) # DEBUG
    print("[INFO] RMSE: %.3f" % rmse) # DEBUG
    print("[INFO] R2_Score: %.3f" % r2) # DEBUG
    print("[INFO] MAE: %.3f" % mae) # DEBUG 
    
    # plot the results
    plt.plot(test)
    plt.plot(predictions, color='red')
    
    plt.xticks(np.arange(0, len(predictions)+1, 5))
    plt.xlabel("Days (transformed to be continous)")
    plt.ylabel("Difference on the # of users")
    plt.title("Antenna %i: Usage pattern of connected users per day\n(RMSE / R2 / MAE: %f / %f / %f)" % (antenna_id, rmse, r2, mae))
    plt.legend(["Test data", "Predicted data"])
    plt.grid(True, axis='y', linestyle='--')
    # plt.show() # DEBUG
    plt.savefig("./Tasks/Task7-Results/results/plots/connected_users_per_day-antenna_%i.png" % antenna_id)
    plt.close("all")
    
    return metrics


def main():
    # creates some vars
    usage_days_threshold = 100 # limits the number of days with usage for the antenna to be considered able
    df_antenas = [] # antennas and their features

    start_time1 = time.time()
    data = load_data() # loads the data set
    end_time1 = time.time()
    
    start_time2 = time.time()
    data = preprocessing(data) # preprocess the data
    end_time2 = time.time()
    
    start_time3 = time.time()
    for _, df in data.groupby(["latitude", "longitude"]):
        df_antenas.append(df.reset_index().drop(["index","latitude","longitude"],axis=1))
    end_time3 = time.time()

    antenna_number = 0 # creates the antenna ids
    df_stats_columns = ["antenna_id", "rmse", "r2_score", "mae"]
    statistical_metrics = pd.DataFrame(columns=df_stats_columns)
    start_time4 = time.time()
    for df in df_antenas[:]:
        if len(df)> usage_days_threshold: # only takes the antenas with more than 'usage_days_threshold' usage days
            metrics_of_each_antenna = time_series_prediction(df.copy(), antenna_number)
            new_data = [antenna_number] + list(metrics_of_each_antenna)
            statistical_metrics.loc[len(statistical_metrics)] = new_data
        antenna_number += 1
    statistical_metrics["antenna_id"] = statistical_metrics["antenna_id"].astype(int)
    statistical_metrics.set_index("antenna_id", inplace=True)
    statistical_metrics.to_csv("./Tasks/Task7-Results/results/statistical_metrics.csv")
    end_time4 = time.time()

    time_loading = datetime.timedelta(seconds=end_time1 - start_time1)
    time_preprocessing = datetime.timedelta(seconds=end_time2 - start_time2)
    time_grouping = datetime.timedelta(seconds=end_time3 - start_time3)
    time_tp = datetime.timedelta(seconds=end_time4 - start_time4)
    time_total = time_loading + time_preprocessing + time_grouping + time_tp

    print("\n[INFO] Total exec time statistics") # DEBUG
    print("[INFO] Time spent loading data (h:mm:ss): ", time_loading)
    print("[INFO] Time spent preprocessing data (h:mm:ss): ", time_preprocessing)
    print("[INFO] Time spent grouping the antennas (h:mm:ss): ", time_grouping)
    print("[INFO] Time spent training and predicting (h:mm:ss): ", time_tp)
    print("[INFO] Total exec time (h:mm:ss): ", time_total)
    
    stats_exec_time_header = ["loading", "processing", "grouping", "predicting", "total"]
    exec_time_list = [time_loading, time_preprocessing, time_grouping, time_tp, time_total]
    # saves the time results
    with open('./Tasks/Task7-Results/results/execution_time_in_seconds.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(stats_exec_time_header)
        write.writerow(exec_time_list)

# Calls the main functionality
if __name__ == "__main__":
    main()