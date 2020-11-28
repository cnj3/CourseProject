import datetime


# Read in IEMPrices.txt and return a dictionary of date object keys and the iem normalized price values
def get_iem_data():
    iem_daily_prices = dict()
    with open('data/IEMPrices.txt') as iem_txt:
        lines = iem_txt.readlines()
        for line in lines:
            line = line.strip()
            split_line = line.split('\t')
            date = datetime.datetime.strptime(split_line[0], "%m/%d/%Y").date()
            price = float(split_line[1])
            iem_daily_prices[date] = price
    return iem_daily_prices


if __name__ == '__main__':
    print(get_iem_data())