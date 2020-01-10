# This file runs each model and shows the results.

import decision_tree
import k_nearest_neightbors
import naive_bayes
import utility
import read_data
import accuracy


# *********************************************************************************************************************
# The main function, all functionality and function calls are from here.
def main():
    # ******************************** Part II Hardcoded train.txt and test.txt files *********************************
    # *****************************************************************************************************************
    # *****************************************************************************************************************
    # ******************************** Read data from files ***********************************************************
    # Get the required data from the file.
    train, features, num_of_features = read_data.read_from_train_file('train.txt')
    test = read_data.read_from_test_file('test.txt')
    # Initialize the features.
    utility.create_feature_dictionaries(features, train)
    # ******************************** Decision Tree ******************************************************************
    # Create the model
    tree_model = decision_tree.DecisionTree(num_of_features, utility.all_feature_types)
    # Create the root.
    tree_root = tree_model.create_tree_root(train, list(utility.all_feature_types.keys()),
                                            tree_model.majority_classification(train), 0)
    # Create the tree.
    tree = decision_tree.Tree(tree_root)
    # Run the algorithm on the data set.
    tree_results = tree_model.classify(test, tree)
    # Create the tree string.
    tree_string = tree.create_tree_string(tree_root)
    # Write it to the output.txt file and add a newline.
    with open("output.txt", 'w') as f:
        f.write(tree_string)
        # Separate by newline for accuracy results later.
        f.write('\n')
    # ******************************** KNN ****************************************************************************
    # Create the model.
    knn_model = k_nearest_neightbors.KNearestNeighbors(5, num_of_features)
    # Run the algorithm and get the results.
    knn_results = knn_model.classify(train, test)
    # ******************************** Naives Bayes  ******************************************************************
    # Create the model.
    bayes_model = naive_bayes.NaiveBayes(num_of_features)
    # Run the algorithm and get the results.
    bayes_results = bayes_model.classify(train, test)
    # ******************************** Accuracy  **********************************************************************
    # Call the accuracy output function, send the test set and algorithm results and write results
    # to the output.txt file
    accuracy.accuracy_output(test, tree_results, knn_results, bayes_results, "output.txt")


if __name__ == '__main__':
    main()
