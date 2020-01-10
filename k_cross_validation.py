# Here the data set will be split into k parts, one will be used as the
# test set and the rest will be the training set.

import random


# *********************************************************************************************************************
# Split the data randomly to training set and test set.
def data_cross_validation(k, data):
    # Initialize result lists.
    training_set = list()
    test_set = list()
    # Shuffle the data pseudo-randomly to be consistent during multiple runs.
    random.Random(4).shuffle(data)
    # Split data into chunks.
    data_chunks = [data[i::k] for i in range(k)]
    first = True
    for chunk in data_chunks:
        # Take first chunk as test set.
        if first:
            test_set = chunk
            first = False
        # Take other chunks as training set.
        else:
            training_set = training_set + chunk
    # Return results.
    return training_set, test_set
