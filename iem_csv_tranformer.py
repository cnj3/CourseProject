# This file is a script that should be run once to transform the data in IEMPrices.csv into a simpler form, stored in
# IEMPrices.txt. This file has one line for each date

import csv


def main():
    all_dates = get_all_dates()
    normalized_prices = get_normalized_prices(all_dates)
    write_txt_file(normalized_prices)


# Return a list of all the dates in the data
def get_all_dates():
    with open('data/IEMPrices.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        all_dates = []
        for row in csv_reader:
            date = row[0].replace('∩╗┐', '')
            if date and any(char.isdigit() for char in date) and date not in all_dates:
                all_dates.append(date)
        return all_dates


# Return a list of lists where an inner list has a date and the normalized price for that date
def get_normalized_prices(all_dates):
    with open('data/IEMPrices.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        normalized_prices = []
        for date in all_dates:
            normalized_prices.append(calculate_normalized_price(csv_reader, date))
        return normalized_prices


# For a given date, calculate the normalized price for that date and return it in a list with the date
def calculate_normalized_price(csv_reader, date):
    dem_price = None
    rep_price = None

    for row in csv_reader:
        if row[0] == date:
            row_party = row[1]
            if 'Dem' in row_party:
                dem_price = float(row[2])
            elif 'Rep' in row_party:
                rep_price = float(row[2])
            if dem_price and rep_price:
                break

    normalized_price = dem_price / (dem_price + rep_price)
    return [date, normalized_price]


# Write out all of the dates and the normalized prices for those dates in a txt file
def write_txt_file(normalized_prices):
    with open('data/IEMPrices.txt', 'w+') as txt_file:
        for date_price in normalized_prices:
            output_str = str(date_price[0]) + '\t' + str(date_price[1]) + '\n'
            txt_file.write(output_str)
    print('IEMPrices.txt has been written.s')


if __name__ == '__main__':
    main()
