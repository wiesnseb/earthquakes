import pandas as pd

def import_csv(path, delimiter=',', date_column = 'date'):
    df = pd.read_csv(path, delimiter=delimiter, parse_dates=[date_column])
    print('CSV import successful')
    return df