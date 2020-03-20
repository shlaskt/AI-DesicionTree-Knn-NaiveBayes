from Utils import mode_labels
import heapq

class Model:
    '''
    implementation of Knn algorithm
    for every sample - calc the hamming distance and get the k nearest predictions
    get the mode for the predict
    '''
    def __init__(self, k=5):
        self.data_set = None
        self.data_labels = None
        self.label_key = None
        self.k = k

    def set_data(self, x, y, additional_data):
        self.data_set = x
        self.data_labels = y
        self.label_key = additional_data[0]

    def calc_hamming_distance(self, query, data_sample):
        # count the number of features that are not the same in the 2 samples
        distance = {key: query[key] for key in query if key in data_sample and query[key] != data_sample[key] and key != self.label_key}
        return len(distance)

    def predict(self, test_set):
        '''
        get predictions for the test set
        for every sample - calc the hamming distance and get the k that nearest to the sample
        '''
        predictions = []
        for test_sample in test_set:
            distances = []
            for data_sample, label in zip(self.data_set, self.data_labels):
                distance = self.calc_hamming_distance(test_sample, data_sample)
                distances.append([distance, label])
            # sort by the distance from the smallest and take the k nearest
            # k_nearest_distances = sorted(distances, key=lambda pair: pair[0])[:self.k]
            k_nearest_distances = heapq.nsmallest(self.k, distances, key=lambda pair: pair[0])
            labels = [pair[1] for pair in k_nearest_distances]  # get the labels only
            predictions.append(mode_labels(labels))  # add the mode to the predictions
        return predictions

    def name(self):
        return 'KNN'