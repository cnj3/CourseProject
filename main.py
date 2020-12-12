import data_reader


def main():
    document_corpus = data_reader.DocumentCorpus('data/edited_articles.txt')
    iem_prices = data_reader.get_iem_prices('data/IEMPrices.txt')


if __name__ == '__main__':
    main()
