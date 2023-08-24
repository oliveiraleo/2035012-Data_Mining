import pandas as pd

''' 
    That's the source code used on the Task 4
    The lines containing the keyword 'DEBUG' were commented out on the final version
    of the code
'''
###############
# Global Vars #
###############
data = []

##############################
# Useful auxiliary functions #
##############################
def load_data():
    print("[INFO] Loading the input data... ", end='')
    data = pd.read_csv("./Tasks/Task4-Clustering/task4-data.csv")
    # print(database_transactions) #DEBUG
    print("[ OK ]")
    return data

###################
# Other functions #
###################

def remove_performance_feature(dataset):
    """ Deletes the Performance column from the dataset
    NOTE: Input should be always a list"""
    # global data # creates a "link" between the data var of this function and the data global var
    print("[INFO] Removing the \'Performance\' feature from the dataset... ", end='')
    del dataset["Performance"] # removes column 'Performance'
    # print(dataset) #DEBUG
    print("[ OK ]")
    return dataset

def main():
    global data # creates a "link" between the data var of this function and the data global var
    print("[INFO] Starting the script")
    data = load_data()
    data1 = remove_performance_feature(data) #TODO give data1 a meaningful name
    # print(data1) #DEBUG
    data1.to_csv("./Tasks/Task4-Clustering/results/data-without-performance-feature.csv", index=False)
    print("[INFO] Results saved locally. Execution done!")


# Calls the main functionality
if __name__ == "__main__":
    main()