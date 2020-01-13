# Everything related to the decision tree is in this file.

import utility
import math


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
# The TreeNode class.
# A single node in the decision tree.
class TreeNode(object):
    # *****************************************************************************************************************
    # The constructor.
    def __init__(self, node_depth, leaf, child_nodes, feature, classification_default):
        # Depth of current node in the tree.
        self.node_depth = node_depth
        # Boolean for if current node is a leaf of not.
        self.leaf = leaf
        # List of all child nodes of current node.
        self.child_nodes = child_nodes
        # Feature of this node.
        self.feature = feature
        # Default classification of current node.
        self.classification_default = classification_default


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
# The decision tree model class.
# Builds a decision tree based on the decision tree algorithm.
class DecisionTree(object):
    # *****************************************************************************************************************
    # The constructor.
    def __init__(self, num_of_features, all_feature_types):
        # The amount of features.
        self.num_of_features = num_of_features
        # All possible feature types.
        self.feature_dictionary = all_feature_types

    # *****************************************************************************************************************
    # Start classifying the test set using the tree model and return results.
    def classify(self, test, tree_model):
        # Will hold classification results.
        results = []
        # For every example.
        for example in test:
            prediction = []
            # Build the result without the prediction.
            for index in range(self.num_of_features):
                prediction.append((example[index][utility.feature_name], example[index][utility.feature_value]))
            # Predict the classification and add it to the example.
            prediction.append((example[index+1][utility.feature_name], tree_model.predict(prediction,
                                                                                          tree_model.tree_root)))
            # Add final result to the results list.
            results.append(prediction)
        return results

    # *****************************************************************************************************************
    # Creates the root of the tree using the DTL algorithm.
    # Gets the training set, list of features, default classification and depth of the node.
    def create_tree_root(self, train, all_feature_types, default_classification, node_depth):
        # If there are no examples, return the default.
        if len(train) == 0:
            return TreeNode(node_depth, True, None, None, default_classification)
        # If every example has the same classification return it with the tree node.
        elif self.same_classifications(train):
            return TreeNode(node_depth, True, None, None, train[0][self.num_of_features][utility.feature_value])
        # If there are no features return the classification by the majority.
        elif len(all_feature_types) == 0:
            return TreeNode(node_depth, True, None, None, self.majority_classification(train))
        # Otherwise.
        else:
            child_nodes = []
            # Get the optimal feature.
            optimal_feature = self.optimal_feature(all_feature_types, train)
            child_features = all_feature_types[:]
            child_features.remove(optimal_feature)
            # Create a root based on the optimal feature.
            current_tree_root = TreeNode(node_depth, False, child_nodes, optimal_feature, None)
            # Get all possible feature values.
            all_feature_values = sorted(utility.all_feature_types[optimal_feature])
            # Iterate the feature values.
            for value in all_feature_values:
                # Get all examples with the feature set as that value.
                distribution = self.find_distribution_of_feature_value(train, value, optimal_feature)
                # Recursively append child nodes.
                child_nodes.append((value, self.create_tree_root(distribution, child_features, default_classification,
                                                                 node_depth + 1)))
        # Return the result.
        return current_tree_root

    # *****************************************************************************************************************
    # Find the feature with the best distribution.
    def optimal_feature(self, all_feature_types, train):
        # None current optimal feature.
        optimal_feature = None
        # Set the max gain.
        maximum_information_gain = -100
        # Get the entropy.
        entropy = self.calculate_entropy(train)
        # Iterate all the features.
        for feature in all_feature_types:
            # Calculate the information gain from the feature.
            information_gain = self.calculate_information_gain(entropy, train, feature)
            # If it's higher than the current maximum update it.
            if information_gain > maximum_information_gain:
                maximum_information_gain = information_gain
                optimal_feature = feature
        # Return the feature with maximum information gain.
        return optimal_feature

    # *****************************************************************************************************************
    # Return all examples with a feature set as a certain value to find distribution.
    def find_distribution_of_feature_value(self, examples_to_search, value, feature):
        # Will hold the examples.
        distribution = []
        # resolve the index of the desired feature
        index = self.find_feature_index(feature, examples_to_search[0])
        # For every example.
        for example in examples_to_search:
            # If that feature is set to the required value.
            if example[index][utility.feature_value] == value:
                # Add it to the list.
                distribution.append(example)
        # Then return the distribution.
        return distribution

    # *****************************************************************************************************************
    # Check if all the classifications of the examples are the same.
    def same_classifications(self, examples):
        # Get a classification.
        classification = examples[0][self.num_of_features][utility.feature_value]
        # Iterate the examples.
        for example in examples:
            # If found a different classification than the one we marked before return false.
            if example[self.num_of_features][utility.feature_value] != classification:
                return False
        # Otherwise return true.
        return True

    # *****************************************************************************************************************
    # Find the majority in between two different classifications.
    def majority_classification(self, train):
        # Counters for classifications.
        positive_classification = 0
        negative_classification = 0
        # For every example in the training set.
        for example in train:
            # Get the classification.
            classification = example[self.num_of_features][utility.feature_value]
            # If it's positive increase the positive counter.
            if classification == utility.confirm_value('yes'):
                positive_classification += 1
            # Otherwise increase the negative counter.
            else:
                negative_classification += 1
        # Return the classification based on the majority.
        if positive_classification >= negative_classification:
            return utility.confirm_value('yes')
        if positive_classification < negative_classification:
            return utility.confirm_value('no')

    # *****************************************************************************************************************
    # Calculate the information gain.
    def calculate_information_gain(self, entropy, examples, feature):
        summation = 0
        temp = sorted(list(self.feature_dictionary[feature]))
        for feat_val in temp:
            sub_example = self.find_distribution_of_feature_value(examples, feat_val, feature)
            percent = float(len(sub_example)) / float(len(examples))
            ent = self.calculate_entropy(sub_example)
            summation += (percent * ent)
        return entropy - summation

    # *****************************************************************************************************************
    # Calculate the entropy.
    def calculate_entropy(self, train):
        # Find the probability of a positive classification and negative classification.
        positive_classification_probability = self.classification_probability(train,
                                                                              utility.confirm_value('yes'))
        negative_classification_probability = self.classification_probability(train,
                                                                              utility.confirm_value('no'))
        # If either one of the classification's probabilities in 0, return 0.
        if positive_classification_probability == 0 or negative_classification_probability == 0:
            return 0
        # Calculate the entropy and return the result.
        result_entropy = - positive_classification_probability * math.log(positive_classification_probability, 2)\
                         - negative_classification_probability * math.log(negative_classification_probability, 2)
        return result_entropy

    # *****************************************************************************************************************
    # Calculate probability of certain classification.
    def classification_probability(self, train, classification):
        # If there are no training examples, return 0.
        if len(train) == 0:
            return 0
        # Counter used to count classifications found.
        counter = 0
        # For every example, if it has the required classification, increase the counter.
        for example in train:
            if example[self.num_of_features][utility.feature_value] == classification:
                counter += 1
        # Calculate probability by dividing the counter by the amount of examples.
        amount = len(train)
        return counter / amount

    # *****************************************************************************************************************
    # Find the index of a certain feature in an example.
    def find_feature_index(self, feature_to_find, example):
        index = 0
        # Search each feature in the example.
        for feature in example:
            # If the name equals to the feature we want to find return the index.
            if feature[utility.feature_name] == feature_to_find:
                return index
            index += 1


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
# The Tree class.
# Will be run to resolve the tree by the model.
class Tree(object):
    # *****************************************************************************************************************
    # The constructor.
    def __init__(self, tree_root):
        # Sets the root of the tree.
        self.tree_root = tree_root

    # *****************************************************************************************************************
    # Predicts the classification of an example.
    # Gets the example to predict and the root of the tree as params.
    # Returns the node with the correct classification.
    def predict(self, example_to_predict, tree_root):
        # Get the current node.
        current_node = TreeNode(tree_root.node_depth, tree_root.leaf, tree_root.child_nodes, tree_root.feature,
                                tree_root.classification_default)
        # If the current node is a leaf return the default classification.
        if current_node.leaf:
            return current_node.classification_default
        item_found = False
        # Otherwise iterate the child nodes.
        for node in current_node.child_nodes:
            # Go along the decision tree and update the current node.
            value = node[utility.feature_value]
            index = self.find_feature_index(current_node.feature, example_to_predict)
            if node[utility.feature_name] == example_to_predict[index][utility.feature_value]:
                current_node = value
                item_found = True
                break
        # Return the result recursively.
        if item_found:
            return self.predict(example_to_predict, current_node)
        # Otherwise classify based on majority.
        else:
            # Counters.
            positive_classification = 0
            negative_classification = 0
            # Count the positive and negative classifications of the child nodes.
            for node in current_node.child_nodes:
                if node[utility.feature_name].is_leaf and node[utility.feature_name].default == "yes":
                    positive_classification += 1
                elif node[utility.feature_name].is_leaf and node[utility.feature_name].default == "no":
                    negative_classification += 1
            # Return based on majority.
            if positive_classification >= negative_classification:
                return "yes"
            if positive_classification < negative_classification:
                return "no"

    # *****************************************************************************************************************
    # Find the index of a certain feature in an example.
    def find_feature_index(self, feature_to_find, example):
        index = 0
        # Search each feature in the example.
        for feature in example:
            # If the name equals to the feature we want to find return the index.
            if feature[utility.feature_name] == feature_to_find:
                return index
            index += 1

    # *****************************************************************************************************************
    # Builds a string for the decision tree.
    def create_tree_string(self, tree_node):
        # Will hold the result.
        result_tree_string = ""
        # Sort the nodes.
        nodes_sorted = sorted(tree_node.child_nodes)
        # Iterate the nodes from the tree root.
        for node in nodes_sorted:
            result_tree_string += tree_node.node_depth * "\t"
            if tree_node.node_depth > 0:
                result_tree_string += "|"
            # Append the node feature and it's value.
            result_tree_string += tree_node.feature + "=" + node[utility.feature_name]
            current_node = node[utility.feature_value]
            # If the node is a leaf append the classification.
            if current_node.leaf:
                result_tree_string += ":" + current_node.classification_default + "\n"
            # Otherwise keep going recursively.
            else:
                result_tree_string += "\n" + self.create_tree_string(current_node)
        # Return the result.
        return result_tree_string
