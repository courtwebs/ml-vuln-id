from cs613 import *

def selection_of_threshold():
    numpy.random.seed(0)
    bin_data = load_csv("bin_test.csv")
    evt_data = load_csv("evt_test.csv")
    bnb_b_prob_diffs = numpy.array([])
    mnb_b_prob_diffs = numpy.array([])
    mnb_e_prob_diffs = numpy.array([])
    bnb_b_vote_accs = [0] * 8
    mnb_b_vote_accs = [0] * 8
    mnb_e_vote_accs = [0] * 8
    bnb_b_thresh_accs = [0] * 8
    mnb_b_thresh_accs = [0] * 8
    mnb_e_thresh_accs = [0] * 8

    for i in range(NUM_TRIALS):
        print("Executing trial " + str(i+1))
        randomize_data(bin_data)
        randomize_data(evt_data)

        # Partition the data
        # Binary data
        bin_train, bin_valid, bin_test = get_data_partitions(bin_data)
        bin_train_x, bin_train_y, bin_train_bins = bin_train
        bin_valid_x, bin_valid_y, bin_valid_bins = bin_valid
        bin_test_x, bin_test_y, bin_test_bins = bin_test
        # Event data
        evt_train, evt_valid, evt_test = get_data_partitions(evt_data)
        evt_train_x, evt_train_y, evt_train_bins = evt_train
        evt_valid_x, evt_valid_y, evt_valid_bins = evt_valid
        evt_test_x, evt_test_y, evt_test_bins = evt_test

        # Train classifier
        bnb_b = BernoulliNB()
        mnb_b = MultinomialNB()
        mnb_e = MultinomialNB()
        t_bnb_b = bnb_b.fit(bin_train_x, bin_train_y)
        t_mnb_b = mnb_b.fit(bin_train_x, bin_train_y)
        t_mnb_e = mnb_e.fit(evt_train_x, evt_train_y)

        # Validation phase
        bnb_b_prob_diffs = accumulate_prob_diff_bins(t_bnb_b, bnb_b_prob_diffs, bin_valid_x)
        mnb_b_prob_diffs = accumulate_prob_diff_bins(t_mnb_b, mnb_b_prob_diffs, bin_valid_x)
        mnb_e_prob_diffs = accumulate_prob_diff_bins(t_mnb_e, mnb_e_prob_diffs, evt_valid_x)

        # Iteratively apply a threshold between 0 and 0.7 at steps of 0.1. Calculate
        # and save a running total of accuracy for each threshold for plotting the 
        # average later.
        i = 0
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            thresh_acc, voted_acc = do_threshold_prediction(t_bnb_b, bin_valid_x, bin_valid_y, bin_valid_bins, threshold)
            bnb_b_thresh_accs[i] += thresh_acc
            bnb_b_vote_accs[i] += voted_acc

            thresh_acc, voted_acc = do_threshold_prediction(t_mnb_b, bin_valid_x, bin_valid_y, bin_valid_bins, threshold)
            mnb_b_thresh_accs[i] += thresh_acc
            mnb_b_vote_accs[i] += voted_acc

            thresh_acc, voted_acc = do_threshold_prediction(t_mnb_e, evt_valid_x, evt_valid_y, evt_valid_bins, threshold)
            mnb_e_thresh_accs[i] += thresh_acc
            mnb_e_vote_accs[i] += voted_acc

            i += 1

    # Plot the average prob diffs
    print("Bernoulli Naive Bayes, binary data")
    plot_pdiff_hist(bnb_b_prob_diffs)
    print("Multinomial Naive Bayes, binary data")
    plot_pdiff_hist(mnb_b_prob_diffs)
    print("Multinomial Naive Bayes, event data")
    plot_pdiff_hist(mnb_e_prob_diffs)
    # Plot the accuracy curve
    # The accuarcy lists are sums of the accuracy, and need to be divided by NUM_TRIALS.
    for i in range(len(bnb_b_vote_accs)):
        bnb_b_thresh_accs[i] = float(bnb_b_thresh_accs[i]) / float(NUM_TRIALS)
        mnb_b_thresh_accs[i] = float(mnb_b_thresh_accs[i]) / float(NUM_TRIALS)
        mnb_e_thresh_accs[i] = float(mnb_e_thresh_accs[i]) / float(NUM_TRIALS)
        bnb_b_vote_accs[i] = float(bnb_b_vote_accs[i]) / float(NUM_TRIALS)
        mnb_b_vote_accs[i] = float(mnb_b_vote_accs[i]) / float(NUM_TRIALS)
        mnb_e_vote_accs[i] = float(mnb_e_vote_accs[i]) / float(NUM_TRIALS)

    plt.plot(bnb_b_thresh_accs, color='blue')
    plt.plot(mnb_b_thresh_accs, color='green')
    plt.plot(mnb_e_thresh_accs, color='red')
    print("Threshold accuracy curve")
    plt.show()
    plt.cla()
    plt.plot(bnb_b_vote_accs, color='blue')
    plt.plot(mnb_b_vote_accs, color='green')
    plt.plot(mnb_e_vote_accs, color='red')
    print("Voted accuracy curve")
    plt.show()

if __name__ == '__main__':
#    cs613.selection_of_threshold()
    selection_of_threshold()
