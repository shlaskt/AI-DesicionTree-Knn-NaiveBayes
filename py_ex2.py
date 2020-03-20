from Utils import average, split_train_data, split_test_data
import Knn
import NaiveBayes
import DecisionTree
K_CROSS = 5
TRAIN_FILE = './train.txt'
TEST_FILE = './test.txt'
OUTPUT_FILE = './output.txt'


def get_acc(test_y, predictions):
    # get accuracy - get the number of errors and divide by number of samples
    # get 1 - loss
    errors = 0
    for y, y_hat in zip(test_y, predictions):
        if y != y_hat:
            errors += 1
    return 1 - errors / len(test_y)


def cross_data(data):
    '''
    creoss data to K sizes (the last one include the modulo)
    :param data: data to cross
    :return: list of splited lists
    '''
    splited_data = []
    data_len = len(data)
    div_size = data_len // K_CROSS
    res = data_len % K_CROSS
    for i in range(K_CROSS - 1):
        splited_data.append(data[i * div_size: (i + 1) * div_size])
    splited_data.append(data[(K_CROSS - 1) * div_size: K_CROSS * div_size + res])
    return splited_data


def k_fold_cross_validation(x, y):
    '''
    get x and y to do cross validation
    :return: list of tuples, for every k_i:
    ki - (data set without i, data set labels, data test- just i, test labels)
    '''
    test_data = []
    crossed_x = cross_data(x)
    crossed_y = cross_data(y)
    for i in range(K_CROSS):
        test_data.append(
            ([item for array in crossed_x for item in array if array != crossed_x[i]],
             [item for array in crossed_y for item in array if array != crossed_y[i]],
             crossed_x[i],
             crossed_y[i]))
    return test_data


def cross_validation_test(x, y, algorithm, additional_data=None):
    '''
    run the given algorithm k times, every time k-1 parts are data and 1 is the test
    get accuracy for every run from the k times. get the average and return it
    :param additional_data:
    :param x: data samples
    :param y: data labels
    :param algorithm: must have 'set data', 'predict' and 'name' methods
    :param attributes: for the decision tree
    :return: k-times algorithm running accuracies average
    '''
    accuracies = []
    data_test_k_cross_list = k_fold_cross_validation(x, y)
    # every item is a tuple: k_i - (data set without i, data set labels, data test- just i, test labels)
    for data_test_k_cross in data_test_k_cross_list:
        data_set, data_labels, test_set, test_labels = data_test_k_cross
        algorithm.set_data(data_set, data_labels, additional_data)
        prediction = algorithm.predict(test_set)
        accuracy = get_acc(test_labels, prediction)
        print(accuracy)
        accuracies.append(accuracy)
    # get the average of the k times accuracies
    avg_accuracy = average(accuracies)
    print("{0}_accuracy\t{1}".format(algorithm.name(), avg_accuracy))
    return avg_accuracy


def write_accuracies(file_name, accuracies):
    # write the accuracies
    dt, knn, nb = accuracies
    f = open(file_name, "a")  # add to the current output file
    f.write("\n")  # divide from the tree
    f.write("{0}\t{1}\t{2}".format(dt, knn, nb))
    f.close()

# def write_tree(file_name, data, y, additional_data):
#     # write the tree
#     decision_tree = DecisionTree.Model()
#     decision_tree.set_data(data, y, additional_data)
#     d_tree = decision_tree.get_tree()
#     d_tree.write_tree(file_name)


# def handle_data_for_train(dataset):
#     data, y, attributes, label_key = split_train_data(dataset)
#     decision_tree = DecisionTree.Model()
#     decision_tree_accuracy = cross_validation_test(data, y, decision_tree, [label_key, attributes])
#     naive_bayes = NaiveBayes.Model()
#     naive_bayes_accuracy = cross_validation_test(data, y, naive_bayes, [label_key, attributes])
#     knn = Knn.Model()
#     knn_accuracy = cross_validation_test(data, y, knn, [label_key])

def handle_data(train_set, test_set):
    '''
    get the predictions for three algorithms - decision tree, knn and naive bayes
    :param train_set: x
    :param test_set: y
    predict the y_hat, calc the accuracy and write to file the accuracies + the tree
    '''
    # split the files and get the data and labels
    train_data, train_data_labels, attributes, label_key = split_train_data(train_set)
    test_data, test_data_labels = split_test_data(test_set)
    # get the algorithms
    decision_tree, knn, naive_bayes = DecisionTree.Model(), Knn.Model(), NaiveBayes.Model()
    algorithms = [decision_tree, knn, naive_bayes]
    accuracies = []
    # for every algorithm - get the prediction on the test set, calc the accuracy and add to list
    for algorithm in algorithms:
        algorithm.set_data(train_data, train_data_labels, [label_key, attributes])
        prediction = algorithm.predict(test_data)
        accuracy = get_acc(prediction, test_data_labels)
        accuracies.append("{0:.2f}".format(accuracy))  # get the 2 digits after point
    # get the output tree and write to the file
    tree = decision_tree.get_tree()
    tree.write_tree(OUTPUT_FILE)
    # write the accuracies to the same file
    write_accuracies(OUTPUT_FILE, accuracies)


if __name__ == '__main__':
    # handle_data_for_train(train_set)
    handle_data(TRAIN_FILE, TEST_FILE)
