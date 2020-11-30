import datetime
import numpy as np


# Read in IEMPrices.txt and return a dictionary of date object keys and the iem normalized price values
def get_iem_prices():
    iem_prices = dict()
    with open('data/IEMPrices.txt') as iem_txt:
        lines = iem_txt.readlines()
        for line in lines:
            line = line.strip()
            split_line = line.split('\t')
            date = datetime.datetime.strptime(split_line[0], "%m/%d/%Y").date()
            price = float(split_line[1])
            iem_prices[date] = price
    return iem_prices


class DocumentCorpus(object):
    """
    A collection of documents.
    """

    nyt_data_path = None
    documents = []
    document_dates = []
    vocabulary = []

    vocabulary_size = 0
    document_count = 0

    def __init__(self, nyt_data_path):
        """
        Initialize empty document list.
        """
        self.nyt_data_path = nyt_data_path
        self.read_data_file()
        self.build_vocabulary()
        self.vocabulary_size = len(self.vocabulary)
        self.document_count = len(self.documents)

    def read_data_file(self):
        with open('data/articles.txt') as iem_txt:
            lines = iem_txt.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    split_line = line.split('\t')
                    date_str = split_line[0]
                    document_str = split_line[1]
                    document_date = datetime.datetime.strptime(date_str, "%Y %m %d").date()
                    document = document_str.split(' ')
                    self.document_dates.append(document_date)
                    self.documents.append(document)

    def build_vocabulary(self):
        vocabulary_set = set()
        for document in self.documents:
            unique_words_in_document = set(document)
            vocabulary_set = vocabulary_set.union(unique_words_in_document)
        self.vocabulary = list(vocabulary_set)


if __name__ == '__main__':
    print(get_iem_prices())