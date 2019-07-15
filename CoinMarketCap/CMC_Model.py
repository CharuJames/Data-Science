from cmc import coinmarketcap
from datetime import datetime
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import time

def extractData(crypto):
    start_date, end_date = datetime(2018,8,1), datetime(2019,7,7)
    df_cryptos = coinmarketcap.getDataFor(crypto, start_date, end_date, fields = ['Open','High','Close'])
    display(df_cryptos.sample(10))
    return df_cryptos

def dataSplit(df_cryptos):
    train_len = int(len(df_cryptos)*.95)
    test_len = int(len(df_cryptos)*.05)
    train = df_cryptos[:train_len]
    test = df_cryptos[train_len:]
    return train,test

def rrFunction(X,Y,C,AI,index):
    xi = X[index]
    yi = Y[index]
    denom = 1-xi.T@AI@xi
    l =AI+((1/denom)*np.outer(AI@xi.T,xi@AI))
    r =X.T@Y-yi*xi
    w = l@r
    pred = (X@w-Y)**2
    return np.sqrt(sum(pred))

def FindBestC(coin):
    crypto = ['bitcoin','ethereum']
    df_cryptos = extractData(crypto)
    train, test = dataSplit(df_cryptos)
    train_X = train[coin].iloc[:,0:2].values
    train_y = train[coin].iloc[:,2].values
    X = np.array(train_X)
    Y = np.array(train_y)
    XX = X.T@X
    list = dict()
    bestC = 0.01
    for C in [0.01,0.006,0.1,1,10,100]:
        allError = []
        AI = np.linalg.inv(XX + (1/(2*C))*np.identity(X.shape[1]))
        for i in range(0, len(train_X/30)):
            err = rrFunction(X,Y,bestC, AI, i)
            allError.append(err)
            print(str(i)+'/'+str(len(train_X/30)), end='\r')
        list[C] = sum(allError)/len(allError)
        if (list[bestC]>list[C]):
            bestC = C
            print("Best C :", bestC)
        plt.plot(allError, 'b')
        plt.title("Error for "+coin)
        plt.show()
    AI = np.linalg.inv(XX + (1/(2*C))*np.identity(X.shape[1]))
    w = AI@X.T@Y
    
    test_X = test[coin].iloc[:,0:2].values
    test_y = test[coin].iloc[:,2].values
    yh = test_X@w
    
    d1 = (yh-test_y)
    plt.boxplot(d1)
    return w

def predict(data, w):
    pred = data@w
    print("Predicted Closing Value : ", pred)
    
if __name__ == "__main__":
    startTime = time.time()
    coin = input("Enter the coin name (bitcoin, ethereum) for prediction:")
    w = FindBestC(coin)
    Open = input("Enter the Opening price:")
    High = input("Enter the High price:")
    val = np.array([float(Open), float(High)])
    predict(val, w)