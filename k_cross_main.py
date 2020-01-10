# This file runs each model and shows the results using k-fold cross validation.

import decision_tree
import k_nearest_neightbors
import naive_bayes
import utility
import read_data
import accuracy
import k_cross_validation


# *********************************************************************************************************************
# The main function, all functionality and function calls are from here.
def main():
    # ******************************** Part I using k-fold cross validation on the data set  **************************
    # ******************************** Read data from files ***********************************************************
    # Get the required data from the file.
    data_set, features, num_of_features = read_data.read_from_train_file('dataset.txt')
    # ******************************** K cross validation *************************************************************
    # *****************************************************************************************************************
    # *****************************************************************************************************************
    # K cross validation of the data.
    # The data is shuffled and split into k chunks.
    # One chunk is set to be the test set and the rest are mixed to be the training set.
    train, test = k_cross_validation.data_cross_validation(5, data_set)
    # Initialize the features.
    # Cross validation - send the training set.
    utility.create_feature_dictionaries(features, train)
    # ******************************** Decision Tree ******************************************************************
    # Create the model
    tree_model = decision_tree.DecisionTree(num_of_features, utility.all_feature_types)
    # Create the root.
    # Cross validation - send the training set.
    tree_root = tree_model.create_tree_root(train, list(utility.all_feature_types.keys()),
                                            tree_model.majority_classification(train), 0)
    # Create the tree.
    tree = decision_tree.Tree(tree_root)
    # Run the algorithm on the data set.
    # Cross validation - send testing set.
    tree_results = tree_model.classify(test, tree)
    # Create the tree string.
    tree_string = tree.create_tree_string(tree_root)
    # Write it to a file.
    with open("tree.txt", 'w') as f:
        f.write(tree_string)
    # ******************************** KNN ****************************************************************************
    # Create the model.
    knn_model = k_nearest_neightbors.KNearestNeighbors(5, num_of_features)
    # Run the algorithm and get the results.
    # Cross validation - send training and test set.
    knn_results = knn_model.classify(train, test)
    # ******************************** Naives Bayes  ******************************************************************
    # Create the model.
    bayes_model = naive_bayes.NaiveBayes(num_of_features)
    # Run the algorithm and get the results.
    # Cross validation - send training and test set.
    bayes_results = bayes_model.classify(train, test)
    # ******************************** Accuracy  **********************************************************************
    # Call the accuracy function, send the test set and algorithm results to compare and write results to a file.
    accuracy.accuracy(test, tree_results, knn_results, bayes_results)


if __name__ == '__main__':
    main()
