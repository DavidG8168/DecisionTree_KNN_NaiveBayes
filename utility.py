# Some utilities for the rest of the program.

# *********************************************************************************************************************
# The true or false values.
negative = None
positive = None

# *********************************************************************************************************************
# Name of feature, value of feature and label index.
feature_value = 1
feature_name = 0
label = -1

# *********************************************************************************************************************
# Variables.
all_feature_types = {}
num_of_features = None
true_probability = None
false_probability = None


# *********************************************************************************************************************
# Create a dictionary of all options for every feature.
def create_feature_dictionaries(features, training_examples):
    global all_feature_types
    index = 0
    # Iterate the features and create a set for each one of
    # all possible results.
    for feature in features:
        all_options_set = {'d'}
        all_options_set.remove('d')
        for example in training_examples:
            all_options_set.add(example[index][feature_value])
        index += 1
        all_feature_types[feature] = move_set(all_options_set)
        all_options_set.clear()


# *********************************************************************************************************************
# Move set to another set.
def move_set(all_options_set):
    # Copy set to another.
    set_target = {all_options_set.pop()}
    for val in all_options_set:
        set_target.add(val)
    return set_target


# *********************************************************************************************************************
# Initialize the true or false values.
def init_values(value):
    global positive, negative
    # Set the positive/negative values.
    if value in ['yes', 'T', 'True', 'true', "yes", "T", "True", "true"]:
        if positive is None:
            positive = value
            return
    elif value in ['no', 'F', 'False', 'false', "no", "F", "False", "false"]:
        if negative is None:
            negative = value
            return


# *********************************************************************************************************************
# Check if the value is in true or false and return a positive or negative response.
def confirm_value(value):
    # If the value is true return positive result.
    if value in ['yes', 'T', 'True']:
        return positive
    # Otherwise if it is false return negative result.
    elif value in ['no', 'F', 'False']:
        return negative
