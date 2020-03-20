from Tree import Tree
from Utils import get_values_for_attributes, get_relevant_samples, mode_labels
from collections import Counter
import math as math


def has_same_classification(samples_labels):
    # if all the classifications are the same - set on the labels will be in size 1
    return len(set(samples_labels)) == 1


# def shrink_tree(tree):
#     '''
#     shrink the tree - remove subtrees with the same classification
#     '''
#     if len(tree.nodes) == 0:  # leaf
#         return tree.attribute  # classifier
#     # go to the tree depth
#     for node in tree.nodes.values():  # else - continue to the nodes
#         shrink_tree(node)
#     # if all have the same- shrink it, else return the original subtree
#     nodes_attributes = set(map(lambda node: node.attribute, list(tree.nodes.values())))
#     if len(nodes_attributes) == 1:  # all nodes attributes are the same - shrink
#         tree.attribute = list(nodes_attributes)[0]
#         tree.nodes = {}
#     return tree


def get_one_sample_prediction(tree, query):
    '''
    recursive function to find the prediction
    :param tree: current node
    :param query: current sample from test set
    :return: prediction
    '''
    if len(tree.nodes) == 0:  # leaf
        return tree.attribute  # classifier
    value = query[tree.attribute]  # get the value from the current node attribute
    next_tree = tree.nodes[value]  # get the next tree by find where the value lead to
    return get_one_sample_prediction(next_tree, query)  # continue until classification


class Model:
    '''
    implementation of Desicion Tree
    get the data, build tree (DTL algotirhm) and predict the result on test set
    '''
    def __init__(self):
        self.data_set = None
        self.data_labels = None
        self.label_key = None
        self.attributes = None
        self.att_to_values = None
        self.tree = None
        self.default_classification = None

    def set_data(self, data, y, additional_data):
        self.data_set = data
        self.data_labels = y
        self.attributes = additional_data[1]
        self.label_key = additional_data[0]
        self.default_classification = mode_labels(y)
        self.att_to_values = get_values_for_attributes(data, self.attributes)
        self.tree = self.DTL(data, self.attributes, self.default_classification)

    def mode(self, samples):
        # get the mode of y from the specific examples
        labels = Counter(samples)
        # if equal - go with the default classification (mode from all samples)
        most_commons = labels.most_common()
        if len(most_commons) > 1:  # else - all the same
            if most_commons[0][1] == most_commons[1][1]:  # equal - take the default from all the data
                return self.default_classification
        return most_commons[0][0]

    def entropy(self, examples):
        # get the samples entropy
        if len(examples) == 0:
            return 0
        # count the number of times of classifiers in the  current samples
        labels = self.get_y_examples(examples)
        labels_counter = dict(Counter(labels))
        # update dict to be the ratio #times / #samples
        labels_freq = {label: float(times) / len(examples) for label, times in labels_counter.items()}
        # return the entropy
        return sum([-1.0 * p * math.log(p) / math.log(2) if p != 0 else 0 for p in labels_freq.values()])

    def gain(self, samples, attribute):
        # get the gain of the current attribute
        Sv = []
        for value in self.att_to_values[attribute]:
            value_examples = get_relevant_samples(samples=samples, attribute=attribute, value=value)
            attribute_ratio = float(len(value_examples)) / len(samples)
            Sv.append(attribute_ratio * self.entropy(value_examples))
        return self.entropy(samples) - sum(Sv)

    def choose_attributes(self, attributes, samples):
        # create list of [gain for attribute x, attribute x] for every attribute with the given samples
        gains = [[self.gain(samples, attribute), attribute] for attribute in attributes]
        return max(gains, key=lambda gain_att: gain_att[0])[1]  # return the attribute withe the max gain

    def get_y_examples(self, samples):
        # filter the y labels from the curr samples
        return list(map(lambda sample: sample[self.label_key], samples))

    def DTL(self, examples, attributes, default_val):
        '''
        dtl algorithm (ID3) for building decision tree
        '''
        samples_labels = self.get_y_examples(examples)  # get y from data
        if len(examples) == 0:
            return Tree(default_val)
        elif has_same_classification(samples_labels) or len(attributes) == 0:
            return Tree(self.mode(samples_labels))
        else:
            best = self.choose_attributes(attributes, examples)
            tree = Tree(best)
            attributes_values = self.att_to_values[best]
            for value in attributes_values:
                value_examples = get_relevant_samples(samples=examples, attribute=best, value=value)
                subtree = self.DTL(value_examples, [att for att in attributes if att is not best], self.mode(samples_labels))
                #  add a branch to tree with label value and subtree subtree
                tree.insert(subtree=subtree, came_from=value)
            return tree

    # def create_tree(self, examples, attributes, default_val):
    #     # create tree with DTL algorithm and then make it better- shrink unnecessary subtrees
    #     tree_before_shrink = self.DTL(examples, attributes, default_val)
    #     return shrink_tree(tree_before_shrink)

    def predict(self, test_set):
        '''
        for every sample - find the prediction from the tree
        :param test_set: the test data
        :return: predictions for test set
        '''
        predictions = []
        for sample_test in test_set:
            prediction = get_one_sample_prediction(self.tree, sample_test)
            predictions.append(prediction)
        return predictions

    def get_tree(self):
        return self.tree

    def name(self):
        return 'DT'

