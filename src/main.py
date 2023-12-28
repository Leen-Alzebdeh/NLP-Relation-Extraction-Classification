# This file contains the main function for the Naive Bayes Classifier for Text Classification
# TODO: Fix the bug in the cross-validation code, and ensure that the test/training accuracy & prediction is calculated correctly
# TODO: Add smoothing to the word probabilities to avoid zero probabilities
# TODO:
import argparse
import csv
import random
import math
import statistics
from collections import defaultdict


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Naive Bayes Classifier for Text Classification')
    # Path to the training data
    parser.add_argument('--train', type=str, required=True)
    # Path to the test data
    parser.add_argument('--test', type=str, required=True)
    # Path to the output file
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument("-i", "--improved", required=False,
                        action='store_true')

    # Get command line arguments list
    args = parser.parse_args()
    improved = args.improved
    # Read train and test data
    train_data = read_data(args.train, improved)
    test_data = read_data(args.test, improved)
    # Initialize and train the Naive Bayes classifier class
    nb_classifier = NaiveBayesClassifier(improved)

    # 3-fold cross-validation
    folds = cross_validation_split(train_data)

    # Train on each fold and calculate the average accuracy
    accuracies = []
    for i in range(len(folds)):
        # Use one fold for validation, rest for training
        validation_data = folds[i]
        train_fold = []
        for j, fold in enumerate(folds):
            if j != i:
                train_fold.extend(fold)

        nb_classifier = NaiveBayesClassifier(improved)
        nb_classifier.train(train_fold)

        # Validate on the current fold
        correct_count = sum(nb_classifier.predict(
            doc) == label for _, doc, label in validation_data)
        accuracy = correct_count / len(validation_data)
        accuracies.append(accuracy)

    average_accuracy = statistics.mean(accuracies)
    std_accuracy = statistics.stdev(accuracies)
    print(
        f'Mean training Accuracy (3-fold cross-validation): {average_accuracy:.2%}')
    print(
        f'Standard deviation of training Accuracy (3-fold cross-validation): {std_accuracy:.2f}')

    # Train on the entire training set
    nb_classifier = NaiveBayesClassifier(improved)
    nb_classifier.train(train_data)

    # Make predictions on the test set
    predictions = []
    for (id, doc, true_label) in test_data:
        output_label = nb_classifier.predict(doc)
        predictions.append((true_label, output_label, id))

    # Write predictions to the output file
    with open(args.output, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['original_label', 'output_label', 'row_id'])
        writer.writerows(predictions)

    confusion_mtx = create_confusion_mtx(predictions)

    metrics = evaluate_NB(confusion_mtx)

    print("Test accuracy: {:.2f}%".format(metrics[0]*100))
    print(
        f'Micro-precision: {metrics[1][0]:.2%}, micro-recall: {metrics[1][1]:.2%}')
    print(
        f'Macro-precision: {metrics[2][0]:.2%}, macro-recall: {metrics[2][1]:.2%}')


class NaiveBayesClassifier:
    '''
    Naive Bayes Classifier for text classification

    Attributes:
        label_probs: Dictionary of label probabilities
        word_probs: Dictionary of word probabilities
    '''

    def __init__(self, improved):
        '''
        Initializes the classifier with empty dictionaries
        '''
        self.label_probs = defaultdict(float)
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = []
        self.improved = improved

    def train(self, data):
        '''
        Trains the classifier on the given data

        Args:
            data: List of tuples (id, doc, label)

        Returns:
            None
        '''
        total_docs = len(data)  # Total number of documents
        label_counts = defaultdict(int)  # Number of documents in each label
        # Number of documents in each label containing each word
        word_counts = defaultdict(lambda: defaultdict(int))

        # Count the number of documents in each label and the number of documents in each label containing each word
        for _, doc, label in data:
            label_counts[label] += 1
            for word in doc.split():
                word_counts[label][word] += 1
                if word not in self.vocabulary:
                    self.vocabulary.append(word)
        if self.improved:
            # Get the top words based on mutual information scores
            top_words = self.top_mi_selection(
                word_counts, label_counts, total_docs, top_k=0.80)  # Adjust top_k as needed
            # Update vocabulary to include only top words
            self.vocabulary = list(top_words)

        # Calculate the label probabilities
        for label, count in label_counts.items():
            self.label_probs[label] = math.log(count / total_docs)

        # Calculate the word probabilities
        for word in self.vocabulary:
            for label, count in label_counts.items():
                self.word_probs[label][word] = math.log(
                    (word_counts[label][word] + 1) / (count + len(self.vocabulary)))

    def calculate_mutual_information(self, word_counts, label_counts, total_docs):
        """
        Calculates mutual information score for each word across labels.

        Args:
            word_counts (dict): A dictionary of word counts per label.
            label_counts (dict): A dictionary of document counts per label.
            total_docs (int): Total number of documents.

        Returns:
            dict: A dictionary of mutual information scores for each word.
        """
        mutual_information_scores = defaultdict(float)

        for word in set(word for label in word_counts for word in word_counts[label]):
            mi_score = 0
            for label in label_counts:
                # Probability of word and label occurring together
                p_word_label = word_counts[label].get(word, 0) / total_docs
                if p_word_label > 0:
                    # Probability of word
                    p_word = sum(word_counts[l].get(word, 0)
                                 for l in label_counts) / total_docs
                    # Probability of label
                    p_label = label_counts[label] / total_docs

                    mi_score += p_word_label * \
                        math.log(p_word_label / (p_word * p_label))

            mutual_information_scores[word] = mi_score

        return mutual_information_scores

    def top_mi_selection(self, word_counts, label_counts, total_docs, top_k):
        """
        Selects top words based on mutual information scores.

        Args:
            word_counts (dict): A dictionary of word counts per label.
            label_counts (dict): A dictionary of document counts per label.
            total_docs (int): Total number of documents.
            top_k (int): Number of top words to select.

        Returns:
            set: A set containing the top words based on mutual information scores.
        """
        mi_scores = self.calculate_mutual_information(
            word_counts, label_counts, total_docs)
        # Sort words by MI score and get the top k
        num = math.ceil(len(mi_scores.items())*top_k)
        sorted_words = sorted(
            mi_scores.items(), key=lambda x: x[1], reverse=True)[:num]
        return set(word for word, _ in sorted_words)

    def predict(self, doc):
        '''
        Predicts the label of the given document

        Args:
            doc: Document to classify

        Returns:
            Predicted label
        '''
        scores = defaultdict(float)  # Scores for each label

        # Calculate the score for each label
        for label, label_log_prob in self.label_probs.items():
            scores[label] = label_log_prob
            for word in doc.split():
                if word not in self.vocabulary:
                    pass
                else:
                    scores[label] += self.word_probs[label][word]

        # Return the label with the highest score
        return max(scores, key=scores.get)


def read_data(file_path, improved):
    '''
    Reads the data from the given file

    Args:
        file_path: Path to the file

    Returns:
        List of tuples (id, doc, label)
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        # Assuming 'tokens' is the text and 'relation' is the label
        if improved:
            data = [(row[0], row[1], row[2])
                    for row in reader]
        else:
            data = [(row[0], row[1], row[2])
                    for row in reader]
    return data


def add_pos_tags(doc, head_pos, tail_pos):
    sentence = doc.split()
    for idx, word in enumerate(sentence):
        if str(idx) in head_pos.split():
            sentence[idx] = "HEAD_" + word
        elif str(idx) in tail_pos.split():
            sentence[idx] = "TAIL_" + word
    sentence = set(sentence)
    return " ".join(sentence)


def cross_validation_split(data, folds=3):
    '''
    Splits the data into k folds

    Args:
        data: List of tuples (id, doc, label)
        folds: Number of folds

    Returns:
        List of k folds
    '''
    random.shuffle(data)  # Shuffle the data
    fold_size = len(data) // folds
    splits = [data[i:i+fold_size]
              for i in range(0, len(data), fold_size)]  # Split the data into k folds
    return splits


def create_confusion_mtx(predictions):
    '''
      Creates a confusion matrix from the test predictions.
      Args:
          predictions: List of tuples (id, doc, label)

      Returns:
          Confusion matrix
    '''
    confusion_mtx = defaultdict(lambda: defaultdict(int))
    for (true_label, output_label, _) in predictions:
        confusion_mtx[true_label][output_label] += 1

    print(end='\nPredicted\Actual|  ')
    for key, _ in confusion_mtx.items():
        print(key, end='\t')
    print('\n'+'---'*25)
    for key, _ in confusion_mtx.items():
        print(key, end='\t|  ')
        for _, inner_val in confusion_mtx.items():
            print(inner_val[key], end='\t\t')
        print('\n')

    return confusion_mtx


def evaluate_NB(confusion_mtx):
    '''
      Calculates evaluation metrics (precision, (micro and macro) precision, and recall)
      Args:
          confusion_mtx: Dict of dict of confusion matrix

      Returns:
          [accuracy, (micro precision, micro-recall), (macro-precision, macro-recall)]
    '''

    d_met = calc_metrics(confusion_mtx, 'director')
    c_met = calc_metrics(
        confusion_mtx, 'characters')
    p_met = calc_metrics(confusion_mtx, 'performer')
    l_met = calc_metrics(confusion_mtx, 'publisher')

    tp, fn, fp = 0, 0, 0
    pres, rec = 0, 0
    for l in [d_met, c_met, p_met, l_met]:
        pres += l[0]/(l[0]+l[2])
        rec += l[0]/(l[0]+l[1])
        tp += l[0]
        fn += l[1]
        fp += l[2]
    accuracy = (tp) / (sum(d_met))
    micro_pres = tp/(tp+fp)
    micro_rec = tp/(tp+fn)
    macro_pres, macro_rec = pres/4, rec/4

    return [accuracy, (micro_pres, micro_rec), (macro_pres, macro_rec)]


def calc_metrics(confusion_mtx, target):
    '''
      Calculates confusion matrix metrics for each class.
      Args:
          confusion_mtx: Dict of dict of confusion matrix
          target: class.

      Returns:
          [tp, fn, fp, tn]
    '''

    tp = confusion_mtx[target][target]
    # calculating false negative
    fp = sum(confusion_mtx[target].values()) - tp
    # calculating false positive
    fn = 0
    for key, val in confusion_mtx.items():
        if key != target:
            fn += confusion_mtx[key][target]
    tn = 0
    for key, val in confusion_mtx.items():
        if key != target:
            for inner_key, inner_val in val.items():
                if inner_key != target:
                    tn += inner_val
    return [tp, fn, fp, tn]


if __name__ == "__main__":
    main()


# negation code

'''
        for row in reader:
            sentence = row[1].split(" ")
            for negation in negations:
                if negation in sentence:
                    start_index = sentence.index(negation)
                    for i in range(start_index+1, len(sentence)):
                        if sentence[i] in punctuation:
                            break
                        else:
                            sentence[i] = 'NOT_'+sentence[i]

                    row[1] = ' '.join(sentence)
                    print(row[1])
                    break

            data.append((row[0], row[1], row[2]))
'''
