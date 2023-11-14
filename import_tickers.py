import csv
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import django

# Set up the Django environment
sys.path.append('/trading/trading')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading.settings')
django.setup()

from trading_app.models import Ticker, TickerCategory

def import_tickers(csv_file_path):
    ticker_count={'All' : 0, 'Exist' : 0}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print('row:',row)
            symbol = row['\ufeffsymbol'].strip()
            ticker_count['All'] += 1
            # Check if Ticker with this symbol already exists
            if Ticker.objects.filter(symbol=symbol).exists():
                print(f'Skipping existing ticker: {symbol}')
                ticker_count['Exist'] += 1
                continue

            # If the Ticker does not exist, proceed to create it and assign categories
            category_names = [name.strip() for name in row['categories'].split(',')]
            categories = []
            for name in category_names:
                category, created = TickerCategory.objects.get_or_create(name=name)
                categories.append(category)

            ticker = Ticker(
                symbol=symbol,
                company_name=row['company_name']
            )
            ticker.save()  # Save the ticker to generate a primary key for the M2M relationship
            ticker.categories.set(categories)  # Add the categories
            print(f'Created new ticker: {ticker.symbol}, {ticker.company_name}, categories = {ticker.categories}')
    print('Total read tickers:',ticker_count['All'])
    print('Total already existing tickers:', ticker_count['Exist'])
    print('Added', ticker_count['All'] - ticker_count['Exist'], 'tickers.')

if __name__ == '__main__':
    # Check if the script has been given a command-line argument for the CSV path
    if len(sys.argv) < 2:
        print("Usage: python import_tickers.py <path_to_csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]  # Get the CSV file path from the command-line argument
    import_tickers(csv_path)