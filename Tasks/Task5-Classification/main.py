import pandas as pd
import numpy as np
import json
import warnings

# Classifiers Begin
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# Classifiers End

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

# Plots Begin
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
# Plots End

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
# import itertools as its
warnings.filterwarnings("ignore") # Disables some warnings

###############
# Global Vars #
###############
data = [] # creates an empty list

##############################
# Useful auxiliary functions #
##############################
def load_data():
    print("[INFO] Loading the input data... ", end='')
    data = pd.read_csv("./Tasks/Task5-Classification/task5-data.csv")
    # print(data) # DEBUG
    print("[ OK ]")
    return data

# Plot functions

def create_confusion_matrix(y_test, y_result,classifier):
    matriz = confusion_matrix(y_test, y_result)
    # print(matriz) # DEBUG
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classifier.classes_, yticklabels=classifier.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    save_path = "./Tasks/Task5-Classification/results/plots/mean_confusion_matrix_from_last_iteration_" + classifier.__class__.__name__ + ".png"
    plt.savefig(save_path)
    # plt.show() # DEBUG
    plt.close('all')

def plot_confusion_matrice(confusion_matrices,classifier):
    
    sum_matrix = np.zeros_like(confusion_matrices[0])

    # Iterate through the list and add up the matrices
    for matrix in confusion_matrices:
        sum_matrix += matrix
    
    # Calculate the mean matrix
    mean_matrix = sum_matrix / len(confusion_matrices)
    print(f"Mean confusion matrix for the {classifier.__class__.__name__} classifier\n{mean_matrix}") # DEBUG
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_matrix, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classifier.classes_, yticklabels=classifier.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Mean Confusion Matrix - {classifier.__class__.__name__}')
    save_path = "./Tasks/Task5-Classification/results/plots/confusion_matrix_" + classifier.__class__.__name__ + ".png"
    plt.savefig(save_path)
    # plt.show() # DEBUG
    plt.close('all')

def add_median_labels(ax, fmt='.3f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def plot_roc_curve(Y_test, model_probs, classifier_name):
    random_probs = [0 for _ in range(len(Y_test))]
    # calculate AUC
    model_auc = roc_auc_score(Y_test, model_probs)
    # summarize score
    label = classifier_name + ' Model: ROC AUC=%.4f' % (model_auc)
    # print(label) # DEBUG
    # calculate ROC Curve
        # For the Random Model
    random_fpr, random_tpr, _ = roc_curve(Y_test, random_probs, pos_label="BP")
        # For the actual model
    model_fpr, model_tpr, _ = roc_curve(Y_test, model_probs, pos_label="BP")
    # Plot the roc curve for the model and the random model line
    plt.plot(random_tpr, random_fpr, linestyle='--', label='Random')
    # plt.plot(model_fpr, model_tpr, marker='.', label=classifier_name)
    plt.plot(model_tpr, model_fpr, label=label)
    # Create labels for the axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    save_path = "./Tasks/Task5-Classification/results/plots/roc_curve_" + classifier_name + ".png"
    plt.savefig(save_path)
    # plt.show() # DEBUG
    plt.close('all')

# Preprocessing aux functions

def normalize_feature(data, feature_name):
    """Normalizes a feature in the interval [0 , 1]
    NOTE: Var feature_name should always be one str (i.e. feature name to be normalized)
    """
    feature = data[[feature_name]] # creates a df containing only one feature

    try:
        scaler = MinMaxScaler()
        feature_tmp = pd.DataFrame( scaler.fit_transform(feature), columns=feature.columns )
        # print(feature_tmp) # DEBUG
    except(ValueError):
        print("[ERROR] The scaler requires at least two columns to work. Please, check your dataFrame")

    index_to_be_replaced = data.columns.get_loc(feature_name) # gets the column index to be replaced
    data.insert(index_to_be_replaced, "tmp", feature_tmp, True) # adds the new column data to the data
    del data[feature_name] # removes the old column data
    data.rename(columns={"tmp":feature_name}, inplace=True) # renames the tmp column to the old name

def oversample_data(X, y):
    # summarize class distribution
    print("[INFO] Class distribution:", Counter(y))
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority') # 'minority' makes the number of samples equal on both classes
    # oversample = RandomOverSampler(sampling_strategy=0.5) # float number defines the proportion of the # of samples on the underrepresented class compared to the majoritarian one
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)
    # summarize class distribution
    print("[INFO] New class distribution", Counter(y_over))
    return X_over, y_over

def data_preprocessing(data):
    # Preprocessing of the data
    X = data.drop(["Candidate ID", "Performance"], axis=1) # drop candidate ID so it doesn't influence the results
    X = pd.get_dummies(X, dtype=int) # creates the dummies
    y = data["Performance"] # gets a copy of the target results
    
    # transform all the instances of LP and MP on one class - MP_LP
    for i in range(len(y)):
        if y[i] == 'LP' or y[i] == "MP":
            y[i] = 'MP_LP'

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # X_scaled, y = oversample_data(X_scaled, y) # apply an oversampling strategy on the data

    return X_scaled, y

###################
# Other functions #
###################

def classification(X, y, classif, it):
    data_reports = []
    confusion_matrices = []
    probs = []
    
    for _ in range(it): #shuffe the data 30 times and get the data set split
        classifier = classif #Creates another instance of the classifier so there's no interference from the previous results
        # dataset split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle = True) # Split proportion: 70% train, 30% test with shuffle on each iteration
        # apply stratification on the split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle = True, stratify = y) # Split proportion: 70% train, 30% test with shuffle on each iteration
        
        classifier.fit(X_train, y_train)
        y_result = classifier.predict(X_test)
        data_reports.append(classification_report(y_test, y_result, output_dict = True)) # saves the results for each feature
        confusion_matrices.append(confusion_matrix(y_test, y_result))
    
    p = classifier.predict_proba(X_test)[:,1]
    probs = [*probs, *p]
    # print("Probs:", probs) # DEBUG
    # print("Probs length:", len(probs)) # DEBUG
    plot_roc_curve(y_test, probs, classifier.__class__.__name__)

    plot_confusion_matrice(confusion_matrices, classifier)
    #create_confusion_matrix(y_test, y_result,classifier) # creates a confusion matrix for the last iteration only
    return calculate_metrics(data_reports,classifier.__class__.__name__)

def calculate_metrics(report_list,name):
    keys = list(report_list[0].keys())
    keys.remove('macro avg')
    keys.remove('accuracy')
    
    metrics = {}
    metrics_complete = {}
    
    for k in keys:
        metrics[k] = {"precision"   : np.mean([a[k]["precision"] for a in report_list]),
                      "recall"      : np.mean([a[k]["recall"] for a in report_list]),
                      "f1"          : np.mean([a[k]["f1-score"] for a in report_list]),}
        
        # dicts to store the data for the boxplots
        metrics_complete[k] = {"precision"  : [a[k]["precision"] for a in report_list],
                               "recall"     : [a[k]["recall"] for a in report_list],
                               "f1"         : [a[k]["f1-score"] for a in report_list],}
    
    # metrics' boxplots
    for m in metrics_complete: # TODO verificar porque isso aqui só funciona na primeira iteracao
        d = pd.DataFrame(metrics_complete[m])
    
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(data=d,width=0.4)
        # ax = plt.boxplot(d,widths=0.4)
        add_median_labels(ax)
        
        # plt.xlim([0.0, 1.0])
        plt.ylim([0.2, 1.0])
        plt.grid(True, axis='y')
        plt.title(f"{m} Boxplot - {name}")
        save_path = "./Tasks/Task5-Classification/results/plots/metrics_box_plot_" + name + ".png"
        plt.savefig(save_path)
        # plt.show() # DEBUG
    
    plt.close('all')

    return metrics

def main():
    global data # creates a "link" between the data var of this function and the data global var
    iterations = 30 # num of iterations for each classifier
    print("[INFO] Starting the script")
    data = load_data()
    # print(data) # DEBUG
    X, y = data_preprocessing(data)
    classifiers = [DecisionTreeClassifier(),
                   LogisticRegression(),
                   KNeighborsClassifier(),
                   GaussianNB()
                   ]
    errors = {}
    for classifier in classifiers:
        errors[classifier.__class__.__name__] = classification(X, y, classifier, iterations)

    print(json.dumps(errors, indent=4)) # DEBUG
    print("[INFO] Results saved locally and printed above. Execution done!")

# Calls the main functionality
if __name__ == "__main__":
    main()