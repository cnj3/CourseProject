import datetime
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric, \
    strip_multiple_whitespaces, strip_short
from gensim.corpora import Dictionary




# Read in IEMPrices.txt and return a dictionary of date object keys and the iem normalized price values
def get_iem_prices(iem_path):
    iem_prices = dict()
    with open(iem_path) as iem_txt:
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
    document_dct = None
    corpus = None
    dates_dict = {}
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

    def read_data_file(self):
        with open('data/articles.txt') as iem_txt:
            index = 0
            lines = iem_txt.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    split_line = line.split('\t')

                    date_str = split_line[0]
                    # document_date = datetime.datetime.strptime(date_str, "%Y %m %d").date()
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

        self.document_count = len(self.documents)
        self.document_dct = Dictionary(self.documents)
        self.corpus = [self.document_dct.doc2bow(text) for text in self.documents]

    def build_vocabulary(self):
        vocabulary_set = set()
        for document in self.documents:
            unique_words_in_document = set(document)
            vocabulary_set = vocabulary_set.union(unique_words_in_document)
        self.vocabulary = list(vocabulary_set)
        self.vocabulary_size = len(self.vocabulary)
