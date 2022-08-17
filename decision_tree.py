import random  # for splitting the data set into training and testing
from math import log  # for calculating entropy of splits


def train_test_split(dataset, test_size=0.2):  # handles both integer and fractional(proportion) test sizes
    total = range(len(dataset))
    if (type(test_size) == float) and (test_size > 0) and (test_size < 1):  # if fractional value is passed
        size = int(test_size * len(dataset))
        test_index = random.sample(population=total, k=size)
    elif (type(test_size) == int) and (test_size <= len(dataset)):  # if an integer is passed
        test_index = random.sample(population=total, k=test_size)
    else:
        raise Exception('invalid test_size')
    test = [dataset[value] for value in total if value in test_index]
    train = [dataset[value] for value in total if value not in test_index]
    return train, test


def find_unique(dataset, column):  # finding unique entries in a column/feature
    values = set()  # always holds unique values.
    for datapoint in dataset:
        values.add(datapoint[column])
    x = list(values)
    x.sort()  # sorting helps in finding the midpoints
    return x


def get_feature_type(dataset, threshold=10):
    # manual analysis is necessary to determine the threshold before calling this function.
    # This function helps in marking features as continuous and categorical.
    feature_type = []
    for i in range(len(dataset[0]) - 1):
        count = len(find_unique(dataset, i))
        if isinstance(dataset[0][i], str) or count <= threshold:
            feature_type.append('categorical')
        else:
            feature_type.append('continuous')
    return feature_type


def find_centres(dataset):  # Finding the split points for the entire data set
    dict = {}
    for i in range(len(dataset[0]) - 1):  # ignore the labels
        centres = set()  # order is not required and unique centres are needed
        column_values = find_unique(dataset, i)  # find unique feature values
        if feature_type[i] == 'continuous':  # this will be global to this function
            for j in range(1,
                           len(column_values)):  # This entire for loop is just to find mid points. I think this can
                # be avoided--->TO DO
                centres.add((column_values[j - 1] + column_values[j]) / 2)
            dict[i] = centres
        elif len(column_values) > 1:  # categorical variable must have more than one categories. #CHeck the edge case
            for column_value in column_values:
                centres.add(column_value)
            dict[i] = centres

    return dict


def split_data(dataset, feature_index, centre):  # splits the data beased on given feature and given entre point
    left, right = [], []  # the splits
    for datapoint in dataset:
        if feature_type[feature_index] == 'continuous':
            if datapoint[feature_index] <= centre:  # split is not on equality for continuous feature
                left.append(datapoint)
            else:
                right.append(datapoint)
        else:  # categorical
            if datapoint[feature_index] == centre:
                left.append(datapoint)
            else:
                right.append(datapoint)
    return left, right


def class_counts(dataset):  # returns the counts of the data points belong to a class
    counts = {}  # a dictionary of label -> count.
    for datapoint in dataset:
        # in our dataset format, the label is always the last column
        label = datapoint[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def gini(dataset):  # returns the gini value for a data set
    total = len(dataset)
    counts = class_counts(dataset)
    gin = 1
    for label in counts:
        prob = counts[label] / total
        gin -= prob ** 2
    return gin


def entropy(dataset):  # returns the entropy value for a data set
    total = len(dataset)
    counts = class_counts(dataset)
    ent = 0
    for label in counts:
        prob = counts[label] / total
        prod = -prob * log(prob, 2)
        ent += prod
    return ent


def entropy_split(left, right):  # returns entropy for the entire split
    total = len(left) + len(right)
    return (len(left) / total) * entropy(left) + (len(right) / total) * entropy(right)


def gini_split(left, right):  # returns gini value for the entire split
    total = len(left) + len(right)
    return (len(left) / total) * gini(left) + (len(right) / total) * gini(right)


def find_best_split(dataset, split_type):  # function to find the best split point based on the split criteria
    if split_type == 'gini':
        function = gini_split
    else:
        function = entropy_split
    mini_gini = 9999999999
    best_feature_index = '999999999999'
    best_centre = 'X'
    best_left = 'X'
    best_right = 'X'
    potential_split = find_centres(dataset)
    for feature in potential_split:  # for every feature
        for centre in potential_split[feature]:  # for every possible mid point in the feature
            left, right = split_data(dataset, feature, centre)  # CHANGED  # split the data
            current_gini = function(left, right)  # calculate the gini value for the split
            if current_gini < mini_gini:
                mini_gini = current_gini
                best_feature_index = feature
                best_centre = centre
                best_left = left
                best_right = right
    return mini_gini, best_centre, best_feature_index, best_left, best_right


class Decision_Tree:  # class to store the state of the decision tree. Entire tree can be identified if the root is remembered
    def __init__(self, value, question=None, left=None, right=None):
        self.value = value
        self.question = question
        self.left = left
        self.right = right

    def __eq__(self, other):
        if self is None or other is None:
            return False
        return (self.value == other.value) and (self.question == other.question) and (self.left == other.left) and (
                self.right == other.right)


def build_tree(dataset, header=None, split_type='gini', min_samples_split=2, max_depth=None, counter=0,
               featureType=None):  # main function to build the decision tree
    if counter == 0:
        if split_type != 'gini' and split_type != 'entropy':
            raise ValueError("Unsupported split type")
        global feature_type
        if featureType == None:
            feature_type = get_feature_type(dataset)
        else:
            if (len(dataset[0])-1) == len(featureType) and type(featureType) == list:
                feature_type = featureType
            else:
                raise ValueError('Length mismatch between feature type and the data_set')
        if header is not None:  ## block added recently
            if len(header) != len(dataset[0]):
                raise ValueError('Length mismatch between feature type and the header passed')  ##Added recently
    mini_gini, mid, feature_index, left_side, right_side = find_best_split(dataset,
                                                                           split_type)  # find the best split for the database

    if (mini_gini == 0) or (mini_gini == 9999999999) or (left_side == []) or (left_side == 'X') or (
            right_side == []) or (right_side == 'X') or (mid == 'X') or (len(dataset) <= min_samples_split) or (
            (max_depth is not None) and (counter == max_depth)):  # Base cases of no splits
        pred = class_counts(dataset)  # make the prediction and form a leaf node
        # print(predict)
        # print(max(predict,key =predict.get),)
        return Decision_Tree(str(max(pred, key=pred.get)))  ##changing to store string info for leaf
    else:
        counter += 1
        operator = '=='
        if feature_type[feature_index] == 'continuous':  # setting the operator based on the feature type
            operator = '<='
        if header:
            string = "is {} {} {}?".format(header[feature_index], operator, mid)
        else:
            string = "is {} {} {}?".format(feature_index, operator, mid)
        left_tree = build_tree(left_side, header, split_type, min_samples_split, max_depth=max_depth, counter=counter)
        right_tree = build_tree(right_side, header, split_type, min_samples_split, max_depth=max_depth, counter=counter)
        if left_tree == right_tree:  # to handle the case when left sub tree and right sub tree give same prediction
            pred = class_counts(dataset)
            return Decision_Tree(str(max(pred, key=pred.get)))  ## changing to store string value for the leaf node to handle both integer and string labels
        else:
            return Decision_Tree([feature_index, operator, mid], string, left_tree,
                                 right_tree)  # create a node in the tree (not a leaf node)


def print_tree(root, spacing=""):  # inorder traversal of the tree
    # Base case: we've reached a leaf
    if (root.left == None) and (root.right == None):
        print(spacing + "Prediction: ", root.value)
        return

    # Print the question at this node
    print(spacing + str(root.question))

    # Call this function recursively on the left branch
    print(spacing + '--> Left:')
    print_tree(root.left, spacing + "  ")

    # Call this function recursively on the right branch
    print(spacing + '--> Right:')
    print_tree(root.right, spacing + "  ")


def predict(root, test):  # for predicting multiple data points
    predictions = []
    for point in test:
        predictions.append(predict_single(root, point))
    return predictions


def predict_single(root, sample):
    # base case if root.value is a string we actually have reached a prediction.
    if type(root.value) == str:
        return root.value
    else:
        if root.value[1] == '<=':  # prediction based on the operator type
            if sample[root.value[0]] <= root.value[-1]:
                return predict_single(root.left, sample)
            else:
                return predict_single(root.right, sample)
        else:
            if sample[root.value[0]] == root.value[-1]:
                return predict_single(root.left, sample)
            else:
                return predict_single(root.right, sample)
