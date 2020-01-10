# File contains the function used to determine the accuracy of each algorithm.

# Import utility for indexes.
import utility


# *********************************************************************************************************************
# Checks the accuracy of the results from the algorithms.
# Params are the training set, the results from the decision tree
# the results from k_nearest_neighbors and bayes.
# Writes the results to a file.
def accuracy(test_set, tree_results, k_nearest_results, bayes_results):
    # The number of testing examples.
    num_of_examples = len(test_set)
    # The error counters.
    tree_err = 0
    k_nearest_err = 0
    bayes_err = 0
    # Iterate the examples and count each error by comparing results.
    for example, tree_res, k_res, bayes_res in zip(test_set, tree_results, k_nearest_results, bayes_results):
        # Decision tree error.
        if example[utility.label][utility.feature_value] != tree_res[utility.label][utility.feature_value]:
            tree_err += 1
        # K nearest neighbors error.
        if example[utility.label][utility.feature_value] != k_res[utility.label][utility.feature_value]:
            k_nearest_err += 1
        # Naive bayer error.
        if example[utility.label][utility.feature_value] != bayes_res[utility.label][utility.feature_value]:
            bayes_err += 1
    # Calculate the error and convert to a string rounded to two numbers after the decimal points.
    tree_str = str(round(float(1 - tree_err / num_of_examples), 2))
    k_nearest_str = str(round(float(1 - k_nearest_err / num_of_examples), 2))
    bayes_str = str(round(float(1 - bayes_err / num_of_examples), 2))
    # Write the results to a file.
    with open("accuracy.txt", 'w') as f:
        # Concat the string for the result.
        res_string = tree_str + '\t' + k_nearest_str + '\t' + bayes_str
        f.write(res_string)


# *********************************************************************************************************************
# Checks the accuracy of the results from the algorithms.
# Params are the training set, the results from the decision tree
# the results from k_nearest_neighbors, bayes and the file we will write to.
# Appends the results to the file.
def accuracy_output(test_set, tree_results, k_nearest_results, bayes_results, filename):
    # The number of testing examples.
    num_of_examples = len(test_set)
    # The error counters.
    tree_err = 0
    k_nearest_err = 0
    bayes_err = 0
    # Iterate the examples and count each error by comparing results.
    for example, tree_res, k_res, bayes_res in zip(test_set, tree_results, k_nearest_results, bayes_results):
        # Decision tree error.
        if example[utility.label][utility.feature_value] != tree_res[utility.label][utility.feature_value]:
            tree_err += 1
        # K nearest neighbors error.
        if example[utility.label][utility.feature_value] != k_res[utility.label][utility.feature_value]:
            k_nearest_err += 1
        # Naive bayer error.
        if example[utility.label][utility.feature_value] != bayes_res[utility.label][utility.feature_value]:
            bayes_err += 1
    # Calculate the error and convert to a string rounded to two numbers after the decimal points.
    tree_str = str(round(float(1 - tree_err / num_of_examples), 2))
    k_nearest_str = str(round(float(1 - k_nearest_err / num_of_examples), 2))
    bayes_str = str(round(float(1 - bayes_err / num_of_examples), 2))
    # Write the results to a file.
    with open(filename, 'a+') as f:
        # Concat the string for the result.
        res_string = tree_str + '\t' + k_nearest_str + '\t' + bayes_str
        f.write(res_string)
