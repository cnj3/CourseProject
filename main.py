import data_reader
import gensim
from gensim import models
import numpy as np


def main():
    print('Reading Data...')
    nyt_data, iem_prices = read_data()
    print('Data Finished being Read\n')

    print('Running ICTM...\n')
    run_ictm(nyt_data, iem_prices, 10)
    print('ICTM finished in')


def read_data():
    nyt_data = data_reader.NYTData('data/articles.txt')
    iem_prices = data_reader.get_iem_prices('data/IEMPrices.txt')
    print("Document Count: " + str(nyt_data.document_count))
    print("Vocabulary Size: " + str(nyt_data.vocabulary_size))
    return nyt_data, iem_prices


def run_ictm(nyt_data, iem_prices, number_of_topics):

    prior = 0

    print('Creating LDA Model...')
    lda_model = gensim.models.LdaModel(nyt_data.corpus,
                                       id2word=nyt_data.document_dct,
                                       alpha='auto',
                                       eta=prior,
                                       num_topics=number_of_topics,
                                       passes=5)
    topics = lda_model.get_topics()  # Size number_of_topics by doc_count
    print('LDA Model Created\n')

    print('Determining Significant Topics...')
    total_topic_probs_by_date = get_topic_total_probability_by_date(nyt_data, lda_model, number_of_topics)
    significant_topics = find_significant_topics(total_topic_probs_by_date, iem_prices, number_of_topics, nyt_data)
    print('Significant Topics Determined')

    return


def get_topic_total_probability_by_date(nyt_data, lda_model, number_of_topics):
    topic_by_date_totals = np.zeros((nyt_data.dates_count, number_of_topics))

    date_index = 0
    for date in nyt_data.dates_list:
        for document_id in nyt_data.dates_dict[date]:
            topic_probs = lda_model.get_document_topics(nyt_data.corpus)[document_id]
            for topic_prob in topic_probs:
                topic_id = topic_prob[0]
                probability = topic_prob[1]
                topic_by_date_totals[date_index][topic_id] += probability
        date_index += 1

    return topic_by_date_totals


def find_significant_topics(total_topic_probs_by_date, iem_prices, number_of_topics, nyt_data):
    price_list = list(iem_prices.values())
    print(nyt_data.dates_count)
    print(iem_prices)
    print(price_list)
    print(len(price_list))
    comparison_array = np.zeros((nyt_data.dates_count, 2))
    comparison_array[:, 0] = price_list

    print(comparison_array)
    print(comparison_array.size)

    significant_topics = []

    #for topic_index in range(number_of_topics):
    return significant_topics


if __name__ == '__main__':
    main()
