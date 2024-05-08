import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    words = message.lower().split(" ")
    return words


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    word_dict = collections.defaultdict(int)
    for message in messages:
        seen = set()
        for word in get_words(message):
            if word not in seen:
                seen.add(word)
                word_dict[word] += 1
    
    filtered_words = {word: count for word, count in word_dict.items() if count >= 5}
    word_dictionary = {word: idx for idx, word in enumerate(filtered_words)}
    return word_dictionary

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    word_dict = word_dictionary
    columns = len(word_dict) 
    rows = len(messages) 

    word_mat = np.zeros((rows,columns))

    for row_idx, message in enumerate(messages):
        for word in get_words(message):
            if word in word_dict.keys():
                col_idx = word_dict[word]
                word_mat[row_idx][col_idx] += 1
    return word_mat


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    n = len(labels)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    phi_y = sum(labels)/n
    phi_y1 = np.ones(cols)
    phi_y0 = np.ones(cols)
    total_y1 = cols #this is from the laplace assumption that every word has been seen atleast once, to avoid 0- errors
    total_y0 = cols
    for index, row in enumerate(matrix):
            if labels[index] == 1:
                total_y1 += sum(row) #summing entire row to add to total val for all y=1
                phi_y1 += row #numpy vector operations allow, += to count words for phi_y1 and phi_y0 , reducing need for a double for loop
            else:
                total_y0 += sum(row)
                phi_y0 += row

    phi_y1 = phi_y1 / total_y1
    phi_y0 = phi_y0 / total_y0

    return {'phi_y': phi_y, 'phi_y1': phi_y1, 'phi_y0': phi_y0}

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    phi_y = model['phi_y']
    phi_y1 = model['phi_y1']
    phi_y0 = model['phi_y0']

    #computing log probs as suggested in q to avoid underflow problems 
    prior_1 = np.log(phi_y)
    prior_0 = np.log(1-phi_y)
    log_phi_y1 = np.log(phi_y1)
    log_phi_y0 = np.log(phi_y0)

    rows = matrix.shape[0]

    predictions = np.zeros(rows)  #predictions for each example
    for index, row in enumerate(matrix):
        p_y1 = np.dot(row, log_phi_y1) + prior_1 # calcualating the prob of y=1 by finding the dot product between the featues x and prob learned before then adding prior 
        p_y0 = np.dot(row, log_phi_y0) + prior_0 #eq different format as using log 
        #the denomentor p(x) does not need to be calculated as it will be the same for either class however this 
        #does mean mean these are not probs but values proportional to the probs
        if p_y1 > p_y0: #if the value proportional to prob of y=1 given x is greater than  y=0 given x predict 1 and 0 viceversa
            predictions[index] = 1
        else:
            predictions[index] = 0
    return predictions

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    phi_y = model['phi_y']
    phi_y1 = model['phi_y1']
    phi_y0 = model['phi_y0']

    token_measure = np.array(np.log(phi_y1/phi_y0))
    five_largest = np.argsort(token_measure)[-5:]
    five_largest = np.sort(five_largest)
    output = []

    for value in five_largest:
        # Iterate through each value in five_largest
        # Add the first key found with that value to output
        for key in dictionary:
            if dictionary[key] == value:
                output.append(key)
                break 
    return output 
    
def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    
    max_accuracy = 0
    best_radius = 0
    for radius in radius_to_consider:
        pred_labels = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(pred_labels == val_labels)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_radius = radius
    return best_radius

def main():
    train_messages, train_labels = util.load_spam_dataset('data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('data/ds6_test.tsv')
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
