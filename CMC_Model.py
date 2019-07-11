from datetime import datetime
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import time

def extractData(crypto):
    start_date, end_date = datetime(2019,1,1), datetime(2019,7,7)
    df_cryptos = coinmarketcap.getDataFor(crypto, start_date, end_date, fields = ['Open','High','Close'])
    display(df_cryptos.sample(10))
    return df_cryptos

def dataSplit(df_cryptos):
    train_len = int(len(df_cryptos)*.95)
    test_len = int(len(df_cryptos)*.05)
    train = df_cryptos[:train_len]
    test = df_cryptos[train_len:]
    return train,test

def normalization(df):
    scaler = preprocessing.StandardScaler().fit(df.iloc[:,0:2].values)
    Xtrain = scaler.transform(df.iloc[:, 0:2].values)
    df['OPen'] = Xtrain[:,0]
    df['High'] = Xtrain[:,1]
    return df

def rrFunction(X,Y,C,AI,index):
    xi = X[index]
    yi = Y[index]
    denom = 1-xi.T@AI@xi
    l =AI+((1/denom)*np.outer(AI@xi.T,xi@AI))
    r =X.T@Y-yi*xi
    w = l@r
    pred = (X@w-Y)**2
    return np.sqrt(sum(pred))

if __name__ == "__main__":
    allError = []
    startTime = time.time()
    crypto = ['bitcoin','ethereum']
    df_cryptos = extractData(crypto)
    train, test = dataSplit(df_cryptos)
    train_X = train['bitcoin'].iloc[:,0:2].values
    train_y = train['bitcoin'].iloc[:,2].values
    X = np.array(train_X)
    Y = np.array(train_y)
    XX = X.T@X
    list = dict()
    bestC = 0.006
    AI = np.linalg.inv(XX + (1/(2*bestC))*np.identity(X.shape[1]))
    for i in range(0, 100):
        err = rrFunction(X,Y,bestC, AI, i)
        allError.append(err)
    #print(allError)    
    plt.plot(allError, 'b')
    plt.title("Error for Bitcoin")
    plt.show()