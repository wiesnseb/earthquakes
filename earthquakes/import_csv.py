import pandas as pd

def import_csv(path, delimiter=',', date_column = 'date', sort_asc = True):
    df = pd.read_csv(path, delimiter=delimiter, parse_dates=[date_column])
    #df = df.set_index(index_col)

    if sort_asc == True:
        df = df.sort_index()

    print('CSV import successful')
    return df