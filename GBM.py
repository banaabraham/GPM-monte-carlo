import pandas as pd
from sklearn.svm import SVR
import numpy as np
import statistics
from numba import jit
import random
from matplotlib import pyplot as plt

@jit
def get_return_df(stock):
    d = pd.read_csv(stock)
    d.iloc[:] = d.iloc[::-1].values
    svr_lin = SVR(kernel= 'linear', C= 1e3)
    X=np.arange(1,len(d)+1,1.0)
    X=np.reshape(X,(len(X),1))
    y=d['Close'].values
    #calculate expected return
    y_lin = svr_lin.fit(X, y).predict(X)
    e_return = (y_lin[-1]-y_lin[0])/y_lin[0]
    #calculate stdev of returns
    mth_return = []  
    for i in range(len(y)-1,0,-10):
        mth_return.append((y[i]-y[i-10])/y[i-10])
    resiko = statistics.stdev(mth_return)
    return e_return,resiko
@jit
def get_return(stock):
    d = pd.read_csv(stock)
    d.iloc[:] = d.iloc[::-1].values
    svr_lin = SVR(kernel= 'linear', C= 1e3) 
    X=np.arange(1,len(d)+1,1.0)
    X=np.reshape(X,(len(X),1))
    y=d["0"].values
    #calculate expected return
    y_lin = svr_lin.fit(X, y).predict(X)
    e_return = (y_lin[-1]-y_lin[0])/y_lin[0]
    #calculate stdev of returns
    mth_return = []  
    for i in range(len(y)-1,0,-10):
        mth_return.append((y[i]-y[i-10])/y[i-10])
    resiko = statistics.stdev(mth_return)
    return e_return,resiko

def monte_carlo(trials,i_price,k,w,e_return,stdev):
    prices = [[] for i in range(trials)]
    days = w*k
    initial = i_price
    for i in range(len(prices)):
        i_price = initial
        prices[i].append(i_price)
        for t in np.arange(0,days,k):
            prices[i].append(i_price)
            eps = random.uniform(-1.0,1.0)
            d_price = i_price*((e_return*k)+(stdev*eps*t**0.5))
            i_price += d_price
    return prices        

def analyze(prices):            
    predictions = []
    for i in range(len(prices)):
        predictions.append(prices[i][-1])
    mean = np.average(predictions)
    dev = statistics.stdev(predictions)
    return predictions,mean,dev            


def main():    
    b=1
    while b==1:
        file_type = input("Input File Type (Dataframe(df) or list (l)): ")
        if file_type=="df" or file_type=="l":
            b=0
    if file_type=="df":
        file = input("input file: ")
        something = get_return_df(file)
    elif file_type=="df":
        file = input("input file: ")
        something = get_return(file)    
                       
    i_price = float(input("Initial price: "))
    e_return,resiko = something
    
    trials = int(input("Number of trials: "))   
    opsi = input("Seconds, Hours, Days (S/H/D): ")
    
    if opsi=="D":
        k=0.04
        hari = float(input("Days : "))
    elif opsi=="S":
        k=1.0
        hari = float(input("Seconds : "))    
    
    prices = monte_carlo(trials,i_price,k,hari,e_return,resiko)
    predictions,mean,dev = analyze(prices)
    
    print("Prediction:")
    print(" Mean: %.2f" %(mean))
    print(" Standard Deviation: %.2f" %(dev))
    print(" price range: %.2f - %.2f" %(mean-dev,mean+dev))
    
    
    hist, bins = np.histogram(predictions, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

if __name__ == '__main__':
    main()
