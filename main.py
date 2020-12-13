import data_reader
import gensim
from gensim import corpora, models


def main():
    print('Reading Data...')
    document_corpus, iem_prices = read_data()
    print('Data Finished being Read\n')

    print('Running ICTM...\n')
    run_ictm(document_corpus, iem_prices, 10, .8)
    print('ICTM finished in')


def read_data():
    document_corpus = data_reader.DocumentCorpus('data/edited_articles.txt')
    iem_prices = data_reader.get_iem_prices('data/IEMPrices.txt')
    print("Document Count: " + str(document_corpus.document_count))
    print("Vocabulary Size: " + str(document_corpus.vocabulary_size))
    return document_corpus, iem_prices


def run_ictm(document_corpus, iem_prices, number_of_topics, mu):
    print('Creating LDA Model...')
    lda_model = gensim.models.LdaModel(document_corpus.corpus,
                                       id2word=document_corpus.document_dct,
                                       alpha='auto',
                                       eta=0,
                                       num_topics=number_of_topics,
                                       passes=5,
                                       decay=mu)
    print('LDA Model Created\ns')

    topics = lda_model.get_topics()
    print(topics)
    print(topics.size)
    print(len(topics))
    print(len(topics[0]))


    return

if __name__ == '__main__':
    main()
