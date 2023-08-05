import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

''' 
    That's the source code used on the Task 3
    The lines containing the keyword 'DEBUG' were commented out on the final version
    of the code
'''
##############################
# Useful auxiliary functions #
##############################
def load_data():
    print("[INFO] Loading the input data... ", end='')
    database_transactions = pd.read_csv("./Tasks/Task3-Association_Rules/task3-data.csv")
    # print(database_transactions) #DEBUG
    print("[ OK ]")
    return database_transactions

###################
# Other functions #
###################

def create_itemsets(input_data):
    support = 0.01

    bool_data = input_data.astype(bool)
    # print(bool_data) #DEBUG
    print("[INFO] Generating the item sets... ", end='')
    itemsets = apriori(bool_data, min_support=support, use_colnames=True)
    # print(itemsets) #DEBUG
    print("[ OK ]")
    print("[INFO] Support:", support) #DEBUG
    print("[INFO] # of item sets:", len(itemsets.index)) #DEBUG
    itemsets.to_csv("./Tasks/Task3-Association_Rules/item_sets_sup-" + str(support) + ".csv", index=False)
    return itemsets

def extract_assoc_rules(itemsets):
    print("[INFO] Extracting association rules... ", end='')
    confidence_threshold = 0.8

    rules = association_rules(itemsets, metric="confidence", min_threshold=confidence_threshold)
    rules.sort_values(by="lift", ascending=False, inplace=True)
    print("[ OK ]")
    print("[INFO] Confidence:", confidence_threshold) #DEBUG
    print("[INFO] # of assoc. rules: ", len(rules.index)) #DEBUG
    rules.to_csv("./Tasks/Task3-Association_Rules/assoc_rules_confidence-" + str(confidence_threshold) +".csv", index=False)

def main():
    print("[INFO] Starting the script")
    d = load_data() # d means data
    i = create_itemsets(d) # i means item sets
    extract_assoc_rules(i)
    print("[INFO] Results saved locally. Execution done!")


# Calls the main functionality
if __name__ == "__main__":
    main()