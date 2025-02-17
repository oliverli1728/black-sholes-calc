import yfinance as yf
import numpy as np

def check_optype(op_type):
    if (op_type not in ['p','c']):
        raise ValueError("Input 'p' for put and 'c' for call!")

def check_trtype(tr_type):
    if (tr_type not in ['b','s']):
        raise ValueError("Input 'b' for Buy and 's' for Sell!")  

def payoff_calculator(x, op_type, strike, op_pr, tr_type, n):
    y=[]
    if op_type=='c':
        for i in range(len(x)):
            y.append(max((x[i]-strike-op_pr),-op_pr))
    else:
        for i in range(len(x)):
            y.append(max(strike-x[i]-op_pr,-op_pr))
    
    y=np.array(y)
    if tr_type=='s':
        y=-y
    return y*n

def check_ticker(ticker):
    """
    Check ticker
    """
    try:
        return yf.Ticker('msft').info['currentPrice']
    except KeyError:
        raise ValueError('Ticker not recognized')

def breakeven(x, y, op_type, strike, op_pr, multi=False):
    idx = 0
    min = 1
    if not multi:
        if op_type == 'c':
            for i in range(len(x)):
                if abs(x[i] - strike - op_pr) < min:
                    min = abs(x[i] - strike - op_pr)
                    idx = i
        else:
            for i in range(len(x)):
                if abs(strike - x[i] - op_pr) < min:
                    min = abs(strike - x[i] - op_pr)
                    idx = i
        return x[idx]
    else:
        for i in range(len(x)):
            if abs(y[i]) < min:
                idx = i
                min = abs(y[i])
        return x[idx]