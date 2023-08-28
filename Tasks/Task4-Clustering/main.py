import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

''' 
    That's the source code used on the Task 4
    The lines containing the keyword 'DEBUG' were commented out on the final version
    of the code
'''
###############
# Global Vars #
###############
data = [] # creates an empty list
KMeans_clusters = 3

##############################
# Useful auxiliary functions #
##############################
def load_data():
    print("[INFO] Loading the input data... ", end='')
    data = pd.read_csv("./Tasks/Task4-Clustering/task4-data.csv")
    # print(database_transactions) #DEBUG
    print("[ OK ]")
    return data

# Preprocessing aux functions

def normalize_feature(data, feature_name):
    """Normalizes a feature in the interval [0 , 1]
    NOTE: Var feature_name should always be one str (i.e. feature name to be normalized)
    """
    feature = data[[feature_name]] # creates a df containing only one feature

    try:
        scaler = MinMaxScaler()
        feature_tmp = pd.DataFrame( scaler.fit_transform(feature), columns=feature.columns )
        # print(feature_tmp) #DEBUG
    except(ValueError):
        print("[ERROR] The scaler requires at least two columns to work. Please, check your dataFrame")

    index_to_be_replaced = data.columns.get_loc(feature_name) # gets the column index to be replaced
    data.insert(index_to_be_replaced, "tmp", feature_tmp, True) # adds the new column data to the data
    del data[feature_name] # removes the old column data
    data.rename(columns={"tmp":feature_name}, inplace=True) # renames the tmp column to the old name

###################
# Other functions #
###################

def remove_performance_feature(dataset):
    """ Deletes the Performance column from the dataset
    NOTE: Input should be always a list"""
    print("[INFO] Removing the \'Performance\' feature from the dataset... ", end='')
    del dataset["Performance"] # removes column 'Performance'
    # print(dataset) #DEBUG
    print("[ OK ]")
    return dataset

def cluster_using_KMeans(input_data, num_clusters):
    # Preprocessing of the data
    sorted_data = input_data.sort_values(input_data.columns[0], ascending=True) # sort the data by Candidate ID...
    del sorted_data["Candidate ID"] # and then remove it so this feature doesn't influence the clustering
    # Apply some extra normalization
    features_to_be_normalized = ["Month of Birth", "Year of Birth", "10th and 12th Completion Years Diff"] # these are the features that weren't normalized before
    for i in features_to_be_normalized:
        normalize_feature(sorted_data, i)
    
    # Create some dummies from the nominal/categorical data and saves the result
    data_with_dummies = pd.get_dummies(sorted_data, columns=['Gender','State (Location)','Degree of study','Specialization in study'], prefix=['gender','state','degree','specialization'])
    data_with_dummies.to_csv("./Tasks/Task4-Clustering/results/data-with-numerical-data-only.csv")

    # Configures and executes K-Means clustering
    kmeans = KMeans(n_clusters = num_clusters, init="random", n_init=10, max_iter=300, random_state=42) # random_state was used to make the results determinist
    kmeans.fit(data_with_dummies)
 
    print("--- K-means results BEGIN ---")
    # print(kmeans.labels_)
    print("Coordinates of the centroids:", kmeans.cluster_centers_) #DEBUG
    print("Total # of iterations:", kmeans.n_iter_) #DEBUG
    print("--- K-means results END ---")


def main():
    global data # creates a "link" between the data var of this function and the data global var
    print("[INFO] Starting the script")
    data = load_data()
    data1 = remove_performance_feature(data) #TODO give data1 a meaningful name
    # print(data1) #DEBUG
    data1.to_csv("./Tasks/Task4-Clustering/results/data-without-performance-feature.csv", index=False)
    cluster_using_KMeans(data1, KMeans_clusters)
    print("[INFO] Results saved locally. Execution done!")


# Calls the main functionality
if __name__ == "__main__":
    main()