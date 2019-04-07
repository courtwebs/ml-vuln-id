import numpy
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import math
from collections import defaultdict
import confusion_matrix as cm
import matplotlib.pyplot as plt

NUM_TRIALS = 20
BNB_BINARY_THRESH = 0.5
MNB_BINARY_THRESH = 0.5
MNB_EVENT_THRESH = 0.5

def load_csv(csvfile):
    data = []

    with open(csvfile) as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = 0

        for row in csv_reader:
            if count < 1:
                count += 1
            else:
                data.append(row)

    data = numpy.array(data)

    return data

"""
Splits a dataset in thirds; training, validate, and test.
"""
def split_data(d):
    split_point = int(math.ceil(float(len(d)) * 1.0/3.0))
    return d[:split_point], d[split_point:split_point*2], d[split_point*2:]

def randomize_data(d):
    numpy.random.shuffle(d)

def calculate_accuracy(predicted, actual):
    correct = 0

    for p, a in zip(predicted, actual):
        if p == a:
            correct += 1

    return float(correct)/float(actual.shape[0])

def print_predictions_stats(predicted, actual):
    cwe_counts = defaultdict(int)

    for p in predicted:
        cwe_counts[p] += 1

    correct = 0

    for p, a in zip(predicted, actual):
        if p == a:
            correct += 1

#    print("Correctly identified " + str(correct) + "/" + str(actual.shape[0]))
#    print("Accuracy: " + str(float(correct)/float(actual.shape[0])))

    l1 = numpy.unique(actual)
    l2 = numpy.unique(predicted)
    labels = numpy.unique(list(set(l1).union(set(l2))))

    return cm.plot_confusion_matrix(actual, predicted, labels), calculate_accuracy(predicted, actual)

def print_probabilities_stats(probabilities):
    p_flattened = probabilities.flatten()
    print("Highest probability ever seen: " + str(p_flattened[numpy.argmax(p_flattened)]))
    print("Lowest probabilities ever seen: " + str(p_flattened[numpy.argmin(p_flattened)]))

    diffs = get_probability_differences(probabilities)
    print("Average difference between in-sample highest and lowest probabilities: " + str(numpy.mean(diffs)))
    print("Standard deviation of in-sample probability differences: " + str(numpy.std(diffs)))

    plt.hist(x=diffs, bins='auto', rwidth=0.85)
    plt.show()

def get_probability_differences(probabilities):
    diffs = []

    for p in probabilities:
        diffs.append(p[numpy.argmax(p)] - p[numpy.argmin(p)])

    diffs = numpy.array(diffs)

    return diffs

def apply_threshold(predicted, actual, binary_names, probabilities, threshold):
    diffs = get_probability_differences(probabilities)
    c_pred = []
    c_act = []
    c_bins = []

    for i in range(len(diffs)):
        if diffs[i] > threshold:
            c_pred.append(predicted[i])
            c_act.append(actual[i])
            c_bins.append(binary_names[i])

    c_pred = numpy.array(c_pred)
    c_act = numpy.array(c_act)
    c_bins = numpy.array(c_bins)

    return c_pred, c_act, c_bins

def do_voting_classification(pred, actual, bins):
    v_bins = []
    v_pred = []
    v_actual = []

    for bin_name in numpy.unique(bins):
        votes = defaultdict(int)
        actual_cwe = 0

        # Calculate the votes
        for i in range(bins.shape[0]):
            if bins[i] == bin_name:
                votes[pred[i]] += 1
                actual_cwe = actual[i]

        max_tally = 0
        max_cwe = 0
        # Figure out which CWE had the most votes
        for cwe, tally in votes.items():
            if tally > max_tally:
                max_cwe = cwe

        v_bins.append(bin_name)
        v_pred.append(max_cwe)
        v_actual.append(actual_cwe)

    return numpy.array(v_pred), numpy.array(v_actual), numpy.array(v_bins)

def reformat_data(d):
    binary_names = d[:,-2:-1]
    data = numpy.delete(d, -2, 1)
    data = data.astype(int)

    return data, binary_names

def filter_data(d, classes_to_keep):
    newdata = []

    for row in d:
        if row[-1] in classes_to_keep:
            newdata.append(row)

    return numpy.array(newdata)

def plot_pdiff_hist(pdiffs):
    plt.bar(x=range(50), height=pdiffs)
    plt.show()
#    plt.xticks(x+0.02, range(0.0,1.0))

def accumulate_prob_diff_bins(trained_nb, prob_diffs, validation_x):
    probabilities = trained_nb.predict_proba(validation_x)
    pdiffs = get_probability_differences(probabilities)
    prob_hist = numpy.histogram(pdiffs, bins=50, range=(0.0, 1.0), density=False)
    prob_hist = prob_hist[0]
    prob_hist = prob_hist.astype(float)

    if prob_diffs.shape[0] == 0:
        prob_diffs = prob_hist
    else:
        prob_diffs += prob_hist
        prob_diffs = prob_diffs / 2

    return prob_diffs

def get_data_partitions(data):
    training, validation, testing = split_data(data)
    training, train_bins = reformat_data(training)
    validation, valid_bins = reformat_data(validation)
    testing, test_bins = reformat_data(testing)

    # Grab our train vectors
    training_x = training[:,:-1]
    training_y = training[:,-1:]
    training_y = training_y.flatten()
    # Grab our validation vectors
    validation_x = validation[:,:-1]
    validation_y = validation[:,-1:]
    validation_y = validation_y.flatten()
    # Grabd our testing vectors
    testing_x = testing[:,:-1]
    testing_y = testing[:,-1:]
    testing_y = testing_y.flatten()

    return (training_x, training_y, train_bins), (validation_x, validation_y, valid_bins), (testing_x, testing_y, test_bins)

def compute_norm_cm(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    return cm

def compute_cm_and_acc(predicted, actual):
    cm = compute_norm_cm(predicted, actual)
    acc = calculate_accuracy(predicted, actual)

    return cm, acc

def threshold_prediction(nb, x, y, bins, threshold):
    probs = nb.predict_proba(x)
    preds = nb.predict(x)
    thresh_pred, thresh_act, thresh_bins = apply_threshold(preds, y, bins, probs, threshold)
    return thresh_pred, thresh_act, thresh_bins

def do_threshold_prediction(nb, x, y, bins, threshold):
    probs = nb.predict_proba(x)
    preds = nb.predict(x)
    thresh_pred, thresh_act, thresh_bins = apply_threshold(preds, y, bins, probs, threshold)
    voted_pred, voted_act, voted_bins = do_voting_classification(thresh_pred, thresh_act, thresh_bins)
    vote_acc = calculate_accuracy(voted_pred, voted_act)
    thresh_acc = calculate_accuracy(thresh_pred, thresh_act)

    return thresh_acc, vote_acc

if __name__ == '__main__':
    print("Please run one of the other scripts: select-threshold.py, random-baseline.py, or test-classifier.py")
