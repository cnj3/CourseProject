import data_reader
import gensim
from gensim import models
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr


def main():
    print('Reading Data...')
    nyt_data, iem_prices = read_data()
    print('Data Finished being Read\n')

    print('Running ICTM...\n')
    run_ictm(nyt_data, iem_prices, 10)
    print('ICTM finished in')

# reads in data from articles.txt and IEMPrices.txt
def read_data():
    nyt_data = data_reader.NYTData('data/articles.txt')
    iem_prices = data_reader.get_iem_prices('data/IEMPrices.txt')
    print("Document Count: " + str(nyt_data.document_count))
    print("Vocabulary Size: " + str(nyt_data.vocabulary_size))
    return nyt_data, iem_prices

# runs LDA algorithm with nyt_data and runs methods to determine the significant topics
def run_ictm(nyt_data, iem_prices, number_of_topics):

    prior = 0

    # creates the LDA Models
    print('Creating LDA Model...')
    lda_model = gensim.models.LdaModel(nyt_data.corpus,
                                       id2word=nyt_data.document_dct,
                                       alpha='auto',
                                       eta=prior,
                                       num_topics=number_of_topics,
                                       passes=5)
    topics = lda_model.get_topics()  # Size number_of_topics by doc_count
    print('LDA Model Created\n')

    # runs methods to determine the signifact topics
    print('Determining Significant Topics...')
    total_topic_probs_by_date = get_topic_total_probability_by_date(nyt_data, lda_model, number_of_topics)
    significant_topics = find_significant_topics(total_topic_probs_by_date, iem_prices, number_of_topics, nyt_data)
    print_topics(significant_topics, lda_model, nyt_data)
    print('Significant Topics Determined\n')

    print('Determining Significant Words....')
    positive_words, negative_words = get_pos_neg_words(iem_prices, nyt_data, significant_topics, lda_model)
    print('Best Gore words: ')
    print_words(positive_words, nyt_data)
    print('Best Bush words: ')
    print_words(negative_words, nyt_data)
    print('Significant Words Determined\n')

    return

# finds the probability of topics by date and creates a multidimensional array 
def get_topic_total_probability_by_date(nyt_data, lda_model, number_of_topics):
    topic_by_date_totals = np.zeros((nyt_data.dates_count, number_of_topics))

    date_index = 0
    # iterates through the list of dates in nyt_data
    for date in nyt_data.dates_list:
        for document_id in nyt_data.dates_dict[date]:
            topic_probs = lda_model.get_document_topics(nyt_data.corpus)[document_id]
            for topic_prob in topic_probs:
                topic_id = topic_prob[0]
                probability = topic_prob[1]
                topic_by_date_totals[date_index][topic_id] += probability
        date_index += 1

    return topic_by_date_totals

# finds the significant topics using the granger algorithm
def find_significant_topics(total_topic_probs_by_date, iem_prices, number_of_topics, nyt_data):
    # initializes multidimensional array with 0's
    comparison_array = np.zeros((nyt_data.dates_count, 2))
    price_list = list(iem_prices.values())
    comparison_array[:, 0] = price_list

    significant_topics = []
    for topic_index in range(number_of_topics):
        comparison_array[:, 1] = total_topic_probs_by_date[:, topic_index]
        # runs granger significance test and appends to list if the test returns a value less than 0.05
        if granger_significance_test(comparison_array) < .05:
            significant_topics.append(topic_index)
    return significant_topics

# runs grainger significance test off of a multidimensional array that is inputted as a parameter
def granger_significance_test(comparison_array):
    max_lag = 7
    smallest_pvalue = 1
    # runs the granger causality test with the array
    results_dict = grangercausalitytests(comparison_array, max_lag, verbose=False)
    for number_of_lags in results_dict.keys():
        this_pvalue = results_dict[number_of_lags][0]['ssr_ftest'][1]
        # looks for smallest p value and assigns it to smallest_pvalue
        if this_pvalue < smallest_pvalue:
            smallest_pvalue = this_pvalue
    return smallest_pvalue

# gets the positive and negative words - 
# states whether the word is a significant word or not
def get_pos_neg_words(iem_prices, nyt_data, significant_topics, lda_model):
    positive_words = []
    negative_words = []
    words_checked = []

    comparison_array = np.zeros((nyt_data.dates_count, 2))
    price_list = list(iem_prices.values())
    comparison_array[:, 0] = price_list

    # iterates through the list of significant topics
    for significant_topic in significant_topics:
        # gets the topic terms
        top_words_in_topic = lda_model.get_topic_terms(significant_topic, 20)
        # iterates through the list of top words in the topic
        for top_word_tuple in top_words_in_topic:
            word_id = top_word_tuple[0]
            comparison_array[:, 1] = nyt_data.word_count_by_date[:, word_id]
            # finds the p value 
            p_val = granger_significance_test(comparison_array)
            # if p value is less than 0.05 and word is not checked
            # uses pearson's correlations on the comparison array and assigns it to correlation value
            if p_val < .05 and word_id not in words_checked:
                correlation, _ = pearsonr(comparison_array[:, 0], comparison_array[:, 1])
                word_tuple = (word_id, p_val, correlation)
                # if correlation is greater than 0, it appends the word to the positive words list
                # else it appends it to the negative words list
                if correlation > 0:
                    positive_words.append(word_tuple)
                else:
                    negative_words.append(word_tuple)
                # appends the word to the list of checked words
                words_checked.append(word_id)

    # sorts the lists of positive and negative words 
    positive_words.sort(key=lambda tup: tup[2], reverse=True)
    negative_words.sort(key=lambda tup: tup[2], reverse=False)

    return positive_words, negative_words

# prints out the topics 
# iterates through the list of topics and gets the topic terms
# then iterates through those and prints the most significant words based on the topic
def print_topics(topics, lda_model, nyt_data):
    for topic in topics:
        words = []
        top_words_in_topic = lda_model.get_topic_terms(topic, 20)
        for top_word in top_words_in_topic:
            word_id = top_word[0]
            words.append(nyt_data.document_dct[word_id])
        print('Topic ' + str(topic) + ' has top words: ' + str(words))

# prints out the words based on signifance
def print_words(word_tuples, nyt_data):
    words = []
    for word_tuple in word_tuples:
        word_id = word_tuple[0]
        word = nyt_data.document_dct[word_id]
        words.append(word)
    print(words)


if __name__ == '__main__':
    main()
