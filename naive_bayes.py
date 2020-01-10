# The Naive Bayes algorithm is implemented in this file.

import utility

# *********************************************************************************************************************
# Positive or negative classification probabilities.
positive_probability = 0
negative_probability = 0


# *********************************************************************************************************************
# *********************************************************************************************************************
# *********************************************************************************************************************
# The Naive Bayes class will classify
# examples based on the Bayes classification algorithm.
class NaiveBayes(object):
    # The constructor.
    # Gets the number of features.
    def __init__(self, num_of_features):
        self.num_of_features = num_of_features

    # Start classifying the examples.
    # Params are the training set and test set.
    def classify(self, train, test):
        # Create the classification dictionary.
        # Examples that are classified as yes and examples that are classified as no.
        classification_dictionary = self.create_classification_dictionary(train)
        results = []
        # Predict result classification for every test example.
        for example in test:
            prediction = []
            # Build the return example without the classification.
            for index in range(self.num_of_features):
                prediction.append((example[index][utility.feature_name], example[index][utility.feature_value]))
            # Predict the classification and add it to the above built example.
            prediction.append((example[index+1][utility.feature_name],
                               self.predict(example, classification_dictionary)))
            # Append the prediction to the result list.
            results.append(prediction)
        # Return the complete results for all test examples.
        return results

    # Predicts an example.
    # Params are the example to predict and the tags.
    def predict(self, example_to_predict, classification_dictionary):
        # Store corresponding values and their probabilities for final classification prediction.
        positive_value_dictionary = {}
        negative_value_dictionary = {}
        # Iterate the features.
        for index in range(self.num_of_features):
            positive_counter = 0
            # Get all possibilities for the feature type.
            feature_options = utility.all_feature_types[example_to_predict[index][utility.feature_name]]
            # Iterate all positive classifications.
            for classification in classification_dictionary["yes"]:
                # If the current feature value is in a positive classification
                # increase the positive counter.
                if example_to_predict[index][utility.feature_value] == classification[index][utility.feature_value]:
                    positive_counter += 1
            # Get that value.
            current_positive_value = example_to_predict[index][utility.feature_value]
            # Add it to dictionary that will store probability of teh value being in a positive classification.
            positive_value_dictionary[current_positive_value] = positive_counter / (len(classification_dictionary["yes"]
                                                                                        ) + len(feature_options))
            # Do the same as above for the negative classifications.
            negative_counter = 0
            # Iterate all negative classifications.
            for classification in classification_dictionary["no"]:
                # Increase counter if feature is in negative classification.
                if example_to_predict[index][utility.feature_value] == classification[index][utility.feature_value]:
                    negative_counter += 1
            # Add that feature with it's probability to the dictionary.
            current_negative_value = example_to_predict[index][utility.feature_value]
            negative_value_dictionary[current_negative_value] = negative_counter / (len(classification_dictionary["no"]
                                                                                        ) + len(feature_options))
        # Calculate the probabilities of the final classification using all gathered data.
        positive_result_probability = self.classification_probability(positive_value_dictionary) * positive_probability
        negative_result_probability = self.classification_probability(negative_value_dictionary) * negative_probability
        # Base final result on which probability is higher.
        if positive_result_probability >= negative_result_probability:
            return utility.confirm_value('yes')
        if positive_result_probability < negative_result_probability:
            return utility.confirm_value('no')

    # Calculate the probability of a classification.
    def classification_probability(self, result_dictionary):
        dictionary_values = result_dictionary.values()
        result = 1
        # Multiply all values in the dictionary.
        for value in dictionary_values:
            result *= value
        # Return the result.
        return result

    # Create two lists of examples that are classified as yes and examples
    # that are classified as no and then put them into a single dictionary.
    def create_classification_dictionary(self, train):
        global positive_probability, negative_probability
        # The final dictionary.
        classification_dictionary = {}
        # Yes and no classification lists.
        yes_classification = []
        no_classification = []
        # Iterate all all the example in the training set.
        for example in train:
            build_example = []
            # Iterate the features to build the example.
            for index in range(self.num_of_features):
                build_example.append((example[index][utility.feature_name], example[index][utility.feature_value]))
            # Check the final classification.
            # If it is positive, add it to the yes classification list.
            if example[index+1][utility.feature_value] == utility.confirm_value('yes'):
                yes_classification.append(build_example)
            # If it is negative, add it to the no classification list.
            elif example[index+1][utility.feature_value] == utility.confirm_value('no'):
                no_classification.append(build_example)
        # Insert the yes and no classification lists to the
        # result dictionary as the values of the yes and no
        # keys for later use.
        classification_dictionary["yes"] = yes_classification
        classification_dictionary["no"] = no_classification
        # Calculate the probability of the result being positive
        # by dividing the length of the yes list by the length of
        # final classification dictionary and then the probability
        # of the result being negative by simply subtracting the
        # positive probability from 1.
        positive_probability = float(len(yes_classification) / (len(yes_classification) + len(no_classification)))
        negative_probability = 1 - positive_probability
        # Return the final classification dictionary.
        return classification_dictionary
