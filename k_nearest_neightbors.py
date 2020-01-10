# The KNN algorithm is implemented in this file.

import utility


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
# The KNN class.
# Classifies examples using the K nearest neighbors algorithm.
# Currently, we will set k as 5 as in the instructions.
class KNearestNeighbors(object):
    # The constructor.
    # Gets the k value and amount of features.
    def __init__(self, k, num_of_features):
        self.k = k
        self.num_of_features = num_of_features

    # Start classifying the examples.
    # Params are the training set and test set.
    def classify(self, train, test):
        results = []
        # Classify every example in the test set.
        for example in test:
            prediction = []
            # Build the return example without the final prediction.
            for index in range(self.num_of_features):
                prediction.append((example[index][utility.feature_name], example[index][utility.feature_value]))
            # Predict classification and append to results.
            prediction.append((example[index + 1][utility.feature_name], self.predict(example, train)))
            results.append(prediction)
        return results

    # Predicts an example.
    # Params are the example to predict and the training set.
    def predict(self, example_to_predict, train):
        hamming_distances = []
        # If there are less examples in the training set than k we can't do anything.
        if len(train) < self.k:
            return
        # We will classify based on similarity to other examples.
        # Find hamming distance for each example in the training set from the example we want to predict.
        for example in train:
            # Get the hamming distance from current example to the example we want to predict.
            num = self.hamming_distance(example_to_predict, example)
            # Appends the distance and the classification.
            hamming_distances.append((num, example[self.num_of_features][utility.feature_value]))
        # Sort the list by hamming distance.
        sorted_hamming_distances = sorted(hamming_distances, key=lambda x: x[0])
        # Counters.
        positive_result = 0
        negative_result = 0
        # Count the k classifications with the smallest hamming distance and decide based on them.
        for index in range(self.k):
            # If the classification is 'yes' increase the positive counter.
            if sorted_hamming_distances[index][utility.feature_value] == utility.confirm_value('yes'):
                positive_result += 1
            # If the classification is 'no' increase the negative counter.
            elif sorted_hamming_distances[index][utility.feature_value] == utility.confirm_value('no'):
                negative_result += 1
        # Choose classification based on which one is larger.
        if positive_result >= negative_result:
            return 'yes'
        if positive_result < negative_result:
            return 'no'

    # Calculates the hamming distance between two examples.
    # Params are two examples to compare.
    def hamming_distance(self, example_one, example_two):
        # Set the starting difference counter to 0.
        diff = 0
        # Iterate all features.
        for index in range(self.num_of_features):
            # If different values, increase the difference counter.
            if example_one[index][1] != example_two[index][1]:
                diff += 1
        # Return the hamming distance.
        return diff
