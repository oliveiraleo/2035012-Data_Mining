import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' 
    That's the source code used on the Task 2
    The lines containing the keyword 'DEBUG' were commented out on the final version
    of the code
'''
###############
# Global Vars #
###############
data = pd.read_csv("./Tasks/Task2-Getting_to_know_your_data/task2-data.csv")
# print(data) #DEBUG

##############################
# Useful auxiliary functions #
##############################
def remove_element_from_series(series, elem):
    "Removes any ocurrences of elem from the series given as input"
    new_series = series.loc[series != elem]
    return new_series

def return_month_number_from_abbreviation(abbreviation):
    "Returns the number of a given abbreviated month name. NOTE: Case sensitive"
    return {
            'JAN': 1,
            'FEB': 2,
            'MAR': 3,
            'APR': 4,
            'MAY': 5,
            'JUN': 6,
            'JUL': 7,
            'AUG': 8,
            'SEP': 9, 
            'OCT': 10,
            'NOV': 11,
            'DEC': 12
    }[abbreviation]

def save_figure_on_disk(plot, file_name):
    "Saves a figure/plot on the results folder"
    file_path = "./Tasks/Task2-Getting_to_know_your_data/results/" + str(file_name)
    print("[INFO] Saving the figure", file_name)
    plot.savefig(file_path)
    print("[INFO] The figure was sucessfully saved at\n[INFO]", file_path)
###################
# Other functions #
###################
def get_some_statistics_from_feature(feature):
    "Prints the minimum, average & maximum values from a given (feature) series or dataframe"
    print("[INFO] Getting some statistics from feature(s)")
    try: # if it's just one feature, takes its name
        print("=> Some statistics for the feature", feature.name)
    except(AttributeError): # if not, then print all of them
        print("=> Some statistics for the features below")
    try:
        print("\n- Avg val:\n", feature.mean())
        print("\n- Max val:\n", feature.max())
        print("\n- Min val:\n", feature.min())
    except(TypeError):
        print("\n[ERROR] Could not obtain statistics probably because the feature's data wasn't (completely) numerical")

def count_diferent_genders():
    "Plots a pie chart graph containing the present genders"
    labels = data["Gender"].unique()
    x = data.groupby(["Gender"]).size()
    # print(x) #DEBUG

    fig = plt.figure(figsize=(3,3), dpi=150) # adjusts the size of the plots
    plt.pie(x, labels = labels, autopct='%1.1f%%')
    plt.title("Genders in the data set")
    plt.tight_layout()
    # plt.show() #DEBUG
    save_figure_on_disk(fig, "genders.pdf")

def count_diferent_locations():
    "Plots a bar graph containing the location data frequency"
    x = data["State (Location)"].sort_values().unique() # sort to give alphabetical order, unique to avoid duplicates
    y = data.groupby(["State (Location)"]).size() # size gives the number of occurrences for each state
    total_instances = y.sum()
    # total_instances = len(data["State (Location)"].index) #TODO investigate if this is fastest for larger data
    # print(x) #DEBUG
    # print(y) #DEBUG
    # print(total_instances) #DEBUG

    y_axis_labels = []
    for i in y: # calculates the percentage for each occurrence
        y_axis_labels.append(float('{:.2f}'.format((i / total_instances)*100))) # stores it on y_axis_labels list

    y_axis_labels = np.sort(y_axis_labels) # sorts the data in order to display it correctly on the plot

    plt.figure(figsize=(8,11), dpi=200) # adjusts the size of the plots
    plt.tight_layout()
    plt.bar(x, y)
    plt.grid(True, axis='y', linestyle=':')
    plt.yticks(np.unique(y), labels=np.unique(y_axis_labels)) # unique was added to avoid printing more than once the same label
    plt.ylabel("Frequency of occurence (%)")
    plt.xlabel("State / Location codes")
    xlocs, xlabs = plt.xticks() # gets some parameters from the xticks
    for i, v in enumerate(y): # writes the labels on top of each column
        plt.text(xlocs[i], v, str(v), horizontalalignment="center", verticalalignment="bottom")
    # plt.show() #DEBUG
    save_figure_on_disk(plt, "locations.pdf")

def count_all_degrees_of_study():
    x = data["Degree of study"]
    # print(x) #DEBUG

    plt.hist(x)
    plt.title("Number of occurences of the types of study degree")
    plt.xlabel("Degree of study")
    plt.ylabel("# of instances")
    plt.yticks(np.arange(0, (x.size)+1, 25))
    plt.show() #TODO adjust the size of the graph
    #TODO save the figures somewhere

def count_degrees_of_study_without_type_X():
    initial_series = data["Degree of study"]
    x = remove_element_from_series(initial_series, 'X')
    # print(x) #DEBUG

    plt.hist(x)
    plt.title("Occurences of study degrees with a low number of instances")
    plt.xlabel("Degree of study")
    plt.ylabel("# of instances")
    plt.yticks(np.arange(0, (x.size)+1))
    plt.show() #TODO adjust the size of the graph
    #TODO save the figures somewhere

def count_relationship_between_degree_and_specialization():
    data.hist("Specialization in study", by="Degree of study") #TODO maybe split the data in 3 diff graphs
    plt.show() #TODO adjust the size of the graph
    #TODO insert some labels on axis to make them more clear to read
    #TODO save the figures somewhere

def remove_leading_spaces_from_column_names():
    data.rename({' Year of Completion of college': 'Year of Completion of college',
                 ' 10th percentage': '10th percentage',
                 ' 12th percentage': '12th percentage',
                 ' College percentage': 'College percentage',
                 ' Year of Completion of college': 'Year of Completion of college',
                 ' English 1': 'English 1'
                 }, axis=1, inplace=True)

def change_months_from_text_to_number():
    # print(data["Month of Birth"]) # DEBUG
    month_abbr = data["Month of Birth"] # gets only the desired column
    for i in range(len(data)):
        data.at[i,"Month of Birth"] = return_month_number_from_abbreviation(month_abbr[i])
    # print(data["Month of Birth"]) # DEBUG

def remove_leading_char_from_column(column_name):
    for i in range(len(data)):
        modified_line_content = data[column_name][i][1:] #removes the leading char
        data.at[i,column_name] = modified_line_content # updates the line

def change_MD_to_average():
    """Replaces the 'MD' (Missing Data) occurrences with the average of the feature on the
    desired columns.
    NOTE: Only works with numerical features"""
    print("[INFO] Replacing the 'MD' occurrences with NaN on candidate skills' data")
    data.replace({"Quantitative Ability 1": 'MD', "Analytical Skills 1": 'MD'}, np.nan , inplace=True) # changes every 'MD' value to NaN on the specified features
    # Format the columns (features) accordingly
    data["Quantitative Ability 1"]=data["Quantitative Ability 1"].astype(float)
    data["Analytical Skills 1"]=data["Analytical Skills 1"].astype(float)
    # Then replaces the NaN values with the feature average
    for i in data.columns.values:
        if i == "Quantitative Ability 1" or i == "Analytical Skills 1":
            avg = data[i].mean(skipna=True)
            print(f"[INFO] Now replacing NaN values of feature {i} with the average {avg}")
            data[i].fillna(avg, inplace=True)
    print("[INFO] Done replacing")

def remove_some_features(features_to_remove):
    """ Deletes columns from a given list of names
    NOTE: Input should be always a list"""
    for i in features_to_remove: # always treat features_to_remove as a list
        del data[i] # removes column 'i'
    # print(data) # DEBUG

def combine_completion_years():
    "TODO"
    combined_data = data["12th Completion year"].astype(int) - data["10th Completion Year"].astype(int)
    # print(combined_data.to_numpy) # DEBUG
    data.insert(9, "10th and 12th Completion Years Diff", combined_data, True)

def preprocess_data():
    "Calls all the auxiliary functions that apply preprocessing techniques on the data"
    change_months_from_text_to_number() # Month is ordinal data in our context, so it makes more sense to use it as numbers instead of text(str)
    remove_leading_spaces_from_column_names() # Some columns had leading spaces on their names, so I removed it
    # Some columns had a character in the beginning of each feature instance, so I removed it
    remove_leading_char_from_column("Year of Birth")
    remove_leading_char_from_column("10th Completion Year")
    remove_leading_char_from_column("12th Completion year")
    remove_leading_char_from_column("Year of Completion of college")
    # print(data[["Candidate ID", "Year of Birth", "10th Completion Year", "12th Completion year", "Year of Completion of college"]]) # DEBUG
    change_MD_to_average() # Except for the last column which will be treated later on
    combine_completion_years() # Records only the difference between the 10th and 12th completion years
    # Removes some features to reduce the data's dimensionality
    features_to_be_removed = ["Name", "Number of characters in Original Name", "10th Completion Year", "12th Completion year"]
    remove_some_features(features_to_be_removed)
    #TODO normalize Year of Completion of college
    #TODO normalize College percentage
    #TODO normalize 10th percentage and 12th percentage ?

def main():
    pass
    # a = data[" 10th percentage"]
    # print(a)
    # get_some_statistics_from_feature(a)
    # get_some_statistics_from_feature(data[["Quantitative Ability 1", "Analytical Skills 1"]]) # ERROR because of str inside these features
                                                                                              # needs to treat that before calling this
    
    # Get some insights regarding the data
    # count_diferent_genders()
    # count_diferent_locations()
    # count_all_degrees_of_study()
    # count_degrees_of_study_without_type_X()
    # count_relationship_between_degree_and_specialization()
    
    # Preprocess the data
    # preprocess_data() # TODO Enable this to preprocess the input data

    # data.to_csv("./Tasks/Task2-Getting_to_know_your_data/results/data-after-preprocessing.csv", index=False) # Saves the data without the index numbers
    
# Calls the main functionallity
if __name__ == "__main__":
    main()