from Utils import get_relevant_samples, mult_list, get_values_for_attributes, mode_labels


def get_attribute_prob(attribute, value, classifier_samples):
    '''
    get the probability when attribute = value in the given classifier_samples
    e.g - Pr(attribute=value | classifier_samples)
    '''
    relevant_samples = get_relevant_samples(classifier_samples, attribute, value)
    att_prob = len(relevant_samples) / len(classifier_samples)
    return att_prob


class Model:
    '''
    implementation of Naive Bayes
    get the data and calc all the predictions from advance
    for every sample in the test - get the prediction for every classifier and get the mode
    '''
    def __init__(self):
        self.data_set = None
        self.data_labels = None
        self.classifiers = None
        self.attributes = None
        self.label_key = None
        self.default_classification = None
        self.att_to_values = None

    def set_data(self, x, y, additional_data=None):
        self.data_set = x
        self.data_labels = y
        self.classifiers = list(set(y))
        self.attributes = additional_data[1]
        self.label_key = additional_data[0]
        self.default_classification = mode_labels(y)
        self.att_to_values = get_values_for_attributes(x, self.attributes)

    def get_classifier_data(self, classifier):
        # get data when label = classifier
        classifier_data = list(filter(lambda x_y: x_y[self.label_key] == classifier, self.data_set))
        return classifier_data

    def create_prob_map(self):
        '''
        create 2 dictionaries for re-use:
        all_data_prob - dict when the keys are - class, attribute, value and map to probability
        labels_prob - the probability for every class in the data
        '''
        all_data_prob = {}
        labels_prob = {}
        for label in self.classifiers:
            filter_label_data = self.get_classifier_data(label)
            labels_prob[label] = len(filter_label_data) / len(self.data_set)
            for attribute in self.attributes:
                for value in self.att_to_values[attribute]:
                    all_data_prob[label, attribute, value] = get_attribute_prob(attribute, value, filter_label_data)
        return all_data_prob, labels_prob

    def predict(self, test_set):
        '''
        get predictions
        for every sample, do the naive bayes algorithm
        :param test_set: data to predict
        :return: predictions - y_hat for every sample in the test set
        '''
        # calc all prediction for every class, attribute, value in advance
        all_data_prob, labels_prob = self.create_prob_map()
        # get prob for every classifier
        predictions = []
        for test_sample in test_set:
            sample_predictions = []  # for more than 2
            for classifier in self.classifiers:
                # get probabilities for every attribute: value in the sample, for every classifier
                sample_prob_for_class = [all_data_prob[classifier, attribute, value] for attribute, value in test_sample.items() if attribute != self.label_key]
                class_prob = mult_list(sample_prob_for_class) * labels_prob[classifier]  # get list multipication * the prob of the class
                sample_predictions.append([classifier, class_prob])
            if sample_predictions[0][1] == sample_predictions[1][1]:  # if the same prediction - take the default
                sample_prediction = self.default_classification
            else:
                sample_prediction = max(sample_predictions, key=lambda preds: preds[1])[0]  # return the max
            predictions.append(sample_prediction)
        # return list of the predictions on the test set
        return predictions

    def name(self):
        return 'naiveBayes'



