from cs613 import *

def get_random_baseline(nb_classifier, data, threshold):
    numpy.random.seed(0)
    nb_acc = 0
    thresh_nb_acc = 0
    vote_nb_acc = 0

    for i in range(NUM_TRIALS):
        print("Executing trial " + str(i+1))
        # Partition the data
        train, valid, test = get_data_partitions(data)
        train_x, train_y, train_bins = train
        valid_x, valid_y, valid_bins = valid
        test_x, test_y, test_bins = test

        randomize_data(train_y)

        # Train classifier
        nb = nb_classifier()
        t_nb = nb.fit(train_x, train_y)

        # Predict without threshold
        nb_pred = t_nb.predict(test_x)

        if i == 0:
            print("Randomized labels: confusion matrix")
            print_predictions_stats(nb_pred, test_y)
   
        nb_acc += calculate_accuracy(nb_pred, test_y)

        thresh_pred, thresh_act, thresh_bins = threshold_prediction(t_nb, test_x, test_y, test_bins, threshold)
        thresh_nb_acc += calculate_accuracy(thresh_pred, thresh_act)

        if i == 0:
            print("Randomized labels with a threshold: confusion matrix")
            print_predictions_stats(thresh_pred, thresh_act)

        # Predict with voting
        voted_pred, voted_act, voted_bins = do_voting_classification(thresh_pred, thresh_act, thresh_bins)
        vote_nb_acc += calculate_accuracy(voted_pred, voted_act)
        if i == 0:
            print("Randomized labels with voting and threshold: confusion matrix")
            print_predictions_stats(voted_pred, voted_act)

    # Calculate average accuracies
    nb_acc = float(nb_acc) / float(NUM_TRIALS)
    thresh_nb_acc = float(thresh_nb_acc) / float(NUM_TRIALS)
    vote_nb_acc = float(vote_nb_acc) / float(NUM_TRIALS)

    print("Randomized labels accuracy: " + str(nb_acc))
    print("Randomized labels and threshold accuract: " + str(thresh_nb_acc))
    print("Randomized labels, threhsold, and voting accuracy: " + str(vote_nb_acc))

if __name__ == '__main__':
    bin_data = load_csv("bin_test.csv")
    evt_data = load_csv("evt_test.csv")

    print("------Bernoulli Naive Bayes With Binary Data------")
    get_random_baseline(BernoulliNB, bin_data, BNB_BINARY_THRESH)
    print("------Multinomial Naive Bayes With Binary Data------")
    get_random_baseline(MultinomialNB, bin_data, MNB_BINARY_THRESH)
    print("------Multinomial Naive Bayes With Event Data------")
    get_random_baseline(MultinomialNB, evt_data, MNB_EVENT_THRESH)
