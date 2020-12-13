import datetime
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric, \
    strip_multiple_whitespaces, strip_short
from gensim.corpora import Dictionary
import numpy as np


# Read in IEMPrices.txt and return a dictionary of date object keys and the iem normalized price values
def get_iem_prices(iem_path):
    iem_prices = dict()
    with open(iem_path) as iem_txt:
        lines = iem_txt.readlines()
        for line in lines:
            line = line.strip()
            split_line = line.split('\t')
            date_str = str(datetime.datetime.strptime(split_line[0], "%m/%d/%Y").date())
            price = float(split_line[1])
            iem_prices[date_str] = price
    return iem_prices


class NYTData(object):
    """
    A collection of documents.
    """

    nyt_data_path = None  # Path to file to read

    documents = []  # List of list. Each inner list is a filtered list of words from a certain document
    document_dct = None  # Special Dictionary object of vocab words and their IDs
    corpus = None  # List of lists. Each inner list is tuples of all words in a given document and their counts
    dates_dict = {}  # Dictionary where date string is key, value is list of doc ids with that date
    dates_list = []  # List of unique dates
    dates_count = 0  # Number of unique dates
    document_count = 0  # Number of documents
    vocabulary_size = 0  # Number of unique vocab terms

    word_count_by_date = None  # 2D np array of size (dates_count, vocabulary_size) with count of each vocab term per day

    def __init__(self, nyt_data_path):
        """
        Initialize empty document list.
        """
        self.nyt_data_path = nyt_data_path
        self.read_data_file()
        self.calculate_word_counts_by_date()

        # print(self.word_count_by_date)
        # print(self.document_dct.num_pos)
        # print(len(self.word_count_by_date))
        # print(self.dates_count)
        # print(self.vocabulary_size)
        # print(len(self.word_count_by_date[0]))


    def read_data_file(self):
        with open('data/articles.txt') as iem_txt:
            index = 0
            lines = iem_txt.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    split_line = line.split('\t')

                    date_str = str(datetime.datetime.strptime(split_line[0], "%Y %m %d").date())
                    if date_str in self.dates_dict:
                        self.dates_dict[date_str].append(index)
                    else:
                        self.dates_dict[date_str] = [index]
                    index += 1

                    document_str = split_line[1]
                    document_str = remove_stopwords(document_str)
                    document_str = strip_numeric(document_str)
                    document_str = strip_punctuation(document_str)
                    document_str = strip_short(document_str)
                    document_str = document_str.strip()
                    document_str = strip_multiple_whitespaces(document_str)
                    document = document_str.split()
                    self.documents.append(document)

        self.dates_list = self.dates_dict.keys()
        self.dates_count = len(self.dates_list)
        self.document_count = len(self.documents)
        self.document_dct = Dictionary(self.documents)
        self.vocabulary_size = len(self.document_dct)
        self.corpus = []
        for document in self.documents:
            self.corpus.append(self.document_dct.doc2bow(document))

    def calculate_word_counts_by_date(self):
        self.word_count_by_date = np.zeros((self.dates_count, self.vocabulary_size))

        date_index = 0
        for date in self.dates_list:
            document_ids = self.dates_dict[date]
            for document_id in document_ids:
                doc_bow = self.corpus[document_id]
                for word_from_bow in doc_bow:
                    word_id = word_from_bow[0]
                    word_count = word_from_bow[1]
                    self.word_count_by_date[date_index][word_id] += word_count
            date_index += 1
