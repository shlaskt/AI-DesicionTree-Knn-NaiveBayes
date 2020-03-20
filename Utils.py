from collections import Counter


def average(lst):
    # get the average number of a list with 2 digits after the point
    avg = sum(lst) / len(lst)
    return "{0:.2f}".format(avg)


def mult_list(lst):
    # multiple the elements in list
    result = 1
    for item in lst:
        result *= item
    return result


def get_values_for_attributes(samples, attributes):
    # return dict with key = attribute and value = possible values
    possible_values = [list(set([sample[att] for sample in samples])) for att in attributes]
    return dict(zip(attributes,possible_values))


def get_relevant_samples(samples, attribute, value):
    '''
    return the samples when the attribute = value in the samples
    '''
    return list(filter(lambda sample: sample[attribute] == value, samples))


# def get_x_to_y_map(x_data, y_data):
#     # create list of lists of [sample , label]
#     return [[x_data[i], y_data[i]] for i in range(len(x_data))]


def mode_labels(labels):
    # get the mode from the labels
    labels_counter = Counter(labels)
    return labels_counter.most_common(1)[0][0]


def split_train_data(file_name):
    '''
    get the data from the dataset
    :param file_name: dataset
    :return: attributes - the fields
            data - list of samples, every sample is a dict of {att1: value, att2: val2...} include classification
            y data - the label for every sample
            class_key - the key for the classifiers column
    '''
    lines_list = [line.rstrip('\n').split('\t') for line in open(file_name)]
    attributes = lines_list[0]
    class_key = lines_list[0][-1]
    data = [dict(zip(attributes, line)) for line in lines_list[1:]]
    y_data = [line[-1] for line in lines_list[1:]]
    attributes = attributes[:-1]
    return data, y_data, attributes, class_key


def split_test_data(file_name):
    '''get the data from the test set
    split the data to list of samples
    every sample is a dict of {att1: value, att2: val2...}
    return the list & the attributes
    '''
    lines_list = [line.rstrip('\n').split('\t') for line in open(file_name)]
    attributes = lines_list[0]
    test_data = [dict(zip(attributes, line)) for line in lines_list[1:]]
    test_data_labels = [line[-1] for line in lines_list[1:]]
    return test_data, test_data_labels
