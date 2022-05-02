import pandas as pd

class pre():
    def preprocess__(self, path,  column_, slice='',  encoding=None, header='infer'):

        df = pd.read_csv(path, encoding=encoding, header=header)

        if slice:
            df = df[slice]

        df.columns = column_

        if (type(df.iloc[:,1][1])==str):
            df.iloc[:,1] = df.iloc[:,1].apply(lambda x: float(x.replace(',', '')))
            
        df['Date'] = df['Date'].str.replace('[^0-9]', '')
        df['Date'] = df['Date'].apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:])
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)

        return df