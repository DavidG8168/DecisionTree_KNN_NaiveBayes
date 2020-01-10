# All functions related to reading the data set from files
# are located here.

import utility


# *********************************************************************************************************************
# Read the data from the training set file.
# Gets the file name as a parameter and returns
# the list of values, list of features and number of features.
def read_from_train_file(filename):
    # Amount of features.
    num_of_features = 0
    # The examples will be placed here.
    examples_list = []
    # Open the file.
    f = open(filename, 'r+')
    # Use to get first line features.
    first = True
    lines = f.read().splitlines()
    # Iterate file lines.
    for line in lines:
        # Take the features from the first line.
        # Split line by tab characters.
        if first:
            tp = line.split("\t")
            # Everything without the label classification.
            features_list = tp[:-1]
            # Length for the number without the label.
            num_of_features = len(tp) - 1
            first = False
        else:
            # Split the line and build the example rows.
            values = line.split("\t")
            row = []
            for val in range(len(values)):
                # Append the feature name with it's value.
                row.append((tp[val], values[val]))
            utility.init_values(values[val])
            # Add the example.
            examples_list.append(row)
    # Return examples, features and their amount.
    return examples_list, features_list, num_of_features


# *********************************************************************************************************************
# Read the data from the test set file.
# Gets the file name as a parameter and returns
# the list of values.
def read_from_test_file(filename):
    # Same as before.
    f = open(filename, 'r+')
    examples_list = []
    lines = f.read().splitlines()
    first = True
    for line in lines:
        if first:
            feature_list = line.split("\t")
            first = False
        else:
            val = line.split("\t")
            row = []
            for i in range(len(val)):
                row.append((feature_list[i], val[i]))
            examples_list.append(row)
    return examples_list
