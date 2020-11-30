import data_reader


def main():
    document_corpus = data_reader.DocumentCorpus('data/IEMPrices.txt')
    iem_prices = data_reader.get_iem_prices()


if __name__ == '__main__':
    main()
