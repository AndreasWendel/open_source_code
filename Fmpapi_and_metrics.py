import csv

from fmp_python.fmp import FMP
import pandas as pd
import plotly.graph_objects as go
import fundamentalanalysis as fa
from plotly.offline import plot
import matplotlib.pyplot as plt
import fredapi as fred
from datetime import timedelta as td
from datetime import datetime as dt
import requests

def getfmp(api,ticker,type):
    api = api
    ticker = ticker
    type = type
    url = (f"https://financialmodelingprep.com/api/v3/{type}/{ticker}?limit=120&apikey={api}")
    data = requests.get(url).json()
    data_formatted = {}
    for value in data:
        date = value['date'][:4]
        del value['date']
        del value['symbol']

        data_formatted[date] = value

    return pd.DataFrame(data_formatted)

def createallfmpcsv(api,ticker,type):

    api = api
    ticker = ticker
    type = type
    for i in type:
        temp = getfmp(api,ticker,i)
        createcsv(temp,ticker,i)



def graphobj(data):
    """
    func for easy graphing with candles
    """
    figure = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['open'],
                                            high=data['high'],
                                            low=data['low'],
                                            close=data['close'])])
    return figure.show()

def createcsv(dataframe,ticker,report):
    """
    :param dataframe: datan
    :param ticker: ticker
    :param report: type of financial report
    folder : creates a folder or enter existing one
    path : path to stored csv
    """
    creater = dataframe
    path = "C:/Users/serig/PycharmProjects/pyquantoptions/csv/"
    folder = ticker+"/"
    report = report+"-"
    ticker = ticker
    type = ".csv"
    fullpath = path+folder+report+ticker+type
    creater.to_csv(fullpath,index=True)

def csvtopandas(ticker,report):
    """
    tar cvs i path och gör den läsbar i python pandas
    index_col=0 ; gör förstra column till index
    report; balalance sheet, cash flow etc
    :param ticker: string för tickern namn
    :return:
    """
    if report == 1:
        report = 'balance-sheet-statement-'+ticker
    elif report == 2:
        report = 'cash-flow-statement-'+ticker
    elif report == 3:
        report = 'financial-statement-full-as-reported-'+ticker
    elif report == 4:
        report = 'income-statement-'+ticker
    else:
        return "invalid" #gör ett error msg
    path = "C:/Users/serig/PycharmProjects/pyquantoptions/csv/"
    ticker = ticker #string
    type = ".csv"
    fullpath = path+ticker+"/"+report+type
    df = pd.read_csv(fullpath, index_col=0)

    return df




#fmpapi key, register at web: https://site.financialmodelingprep.com/developer/docs
apikey = ""
type = ["balance-sheet-statement","cash-flow-statement","income-statement",]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#decide ticker
Ticker = ""

fmp = FMP(api_key=apikey)
fmp.get_quote(Ticker)
tkr_incomestatement = fa.income_statement(Ticker, apikey)


def median_growth(series):
    """
    takes all the yearly data and returns a median for x years
    maybe just use recent 5-10-20 years as a limit

    :param series:
    :return: new dataframe or added idk
    """
    series = series
    temp = pd.DataFrame()
    for i in range(len(series.index)):
        if series.iloc[i] == int or float:
            temp1 = series.iloc[i].sum()
        else:
            temp1 = None
        temp.append(temp1)
    return temp


def CAPM(ticker, key, is_growth, fredapikey_string):
    """ cal capm with isk Free Rate * (beat of stock * Equity Risk Premium)
    risk free rate = gov bond 5 + 10 year / 2 (vill få live data av bonds)
    Equity risk premium =
    ( 1 / (price / pe ratio)) + (dividen yield + gpd growth rate)  / 2 price / pe från fa.quote
    dividend yiled idk, gpd world bank data?
    for div paying stocks use hardcoded damodaran
    otherwise use pe with stocks beta
    Damodaran has the market standard formulations for ERP if you're interested
    bear in mind that works best for dividend paying stocks than non div paying


    """
    method = is_growth
    fredapikey = fredapikey_string
    Fred = fred.Fred(fredapikey)
    weekago = dt.today() - td(days=7)  # less results
    yearago = dt.today() - td(days=360)  # less results
    fiveyear, tenyear = Fred.get_series("DGS5", observation_start=weekago), Fred.get_series("DGS10",
                                                                                            observation_start=weekago)  # same as below
    gdp = Fred.get_series("A191RL1Q225SBEA",
                          observation_start=yearago)  # this takes all from a year ago you only want the latest
    divy = 1.62 * 1.10  # get spy div yield, div / price = yield and then + expected growth
    # gdp is suppose to translate to div growth in some aspect, multpl says 10% growth
    gdp = gdp.iloc[-1]
    profilebeta = fa.profile(ticker, key)
    beta = profilebeta.iloc[2][0] # specifik stock
    temp = fa.quote("spy", apikey)
    pe = temp[0]["pe"]
    if method == False:
        erp = 4.84  # % (Trailing 12 month cash yield) from Damodaran
    else:
        erp = ((1 / pe) * 100)
    # erp = ((1 / pe) * 100) + (divy) / 2  # erp från spy, 1/ pe i sedana hel tal så * 100
    risk_free_rate = (fiveyear.iloc[-1] + tenyear.iloc[-1]) / 2
    discountrate = risk_free_rate + (erp * beta)  # kan vara fel kalkulerad
    # Ra - Rf = Ba * (Rm -Rf)
    # Cost of Equity = Risk-Free Rate of Return + Beta ×
    # (Market Rate of Return– Risk-Free Rate of Return) to reach 1 + 1.1 × (10-1) = 10.9%.
    return discountrate

def pull_data(ticker, key):
    """get stock data
    gör en metod för att automatisk bestäma growth eller div stock
    asigna 5 variable för all när den kallas ie a,b,c,d,e = pull_data()
    """
    ticker = ticker
    key = key
    statement = fa.income_statement(ticker, key)
    enterprise = fa.enterprise(ticker, key)
    cashflow = fa.cash_flow_statement(ticker, key)
    profile = fa.profile(ticker, key)
    quote = fa.quote(ticker, key)
    financial_ratios = fa.financial_ratios(ticker, key)
    n = 4
    temp = 0

    for r in range(n):
        temp = temp + financial_ratios.iloc[52][r]
    # check calc
    if temp/5 > 0.02:
        growth = False
    else:
        growth = True

    drate = discount_rate(ticker, key, growth)
    return statement,enterprise,cashflow,profile,quote ,drate

def pulltemp():
    """get stock data
    gör en metod för att automatisk bestäma growth eller div stock
    """
    statement = 1
    enterprise = 2
    cashflow = 3
    profile = 4
    quote = 5
    return statement,enterprise,cashflow,profile,quote

def present_value(ticker, key,discount_rate):
    """
    present value(pv) = (cashflow1/(1+r)^1) + CFn/(1+r)^n
    n = periods använda alla elr, 5-10 år sedan
    r = discount rate
    CF = cashflow
    används för cashflow increase
    Enterprise Value = Market Cap + Total Debt - Cash & Equivalents - Short-Term Investments
    :param ticker: 
    :param key: 
    :return: 
    """
    statement = fa.cash_flow_statement(ticker,key)
    enterprise = fa.enterprise(ticker,key)
    cashflow = statement.iloc[6]
    n = 5 # current plan only gives 5 years
    r = discount_rate # choose right stock
    r = r/100
    pv = 0
    # cal with net income, wrong valuation maybe for growth prediction instead
    for i in range(n):
         pv = (cashflow[i] / (1 + r) ** i) + pv
    # cal with enterprise
    for i in range(n):
        pv = (enterprise.iloc[6][i] / (1 + r) ** i) + pv
    #ej klar, add current data, new mkt cap as swell with the lastest from api

    return pv


def net_present_value(pv, ii):
    """
    NPV = PV of Cash Flows - Initial Investment
    :param ticker: 
    :param key: 
    :return: 
    """

    npv = pv - ii

    return npv


def IRR_net_present_value(ticker, key):
    """
    Internal Rate of Return (IRR):
    the cash flows are the initial equity investment and the final payout to equity holders at the end, so only cash flows at the beginning and the end
    This formula calculates the discount rate that makes the NPV of an investment opportunity equal to zero. 
    The formula is as follows:
    NPV = CF0 + CF1 / (1 + IRR)^1 + CF2 / (1 + IRR)^2 + ... + CFn / (1 + IRR)^n = 0
    CF0 = Initial cash flow
    CF = Cash flow in each period
    IRR = Internal rate of return
    :param ticker: 
    :param key: 
    :return: 
    """

    balance = fa.balance_sheet_statement(Ticker, apikey)
    start = (len(balance.columns) - 1) # number of years of data, free gives 5
    cf0 = balance.iloc[42][start]
    # cf tar vi som stockholder equity

    npv = 0

    return


def WACC(ticker, key, discount_rate):
    """
    Weighted Average Cost of Capital
    This formula calculates the weighted average cost of capital,
    which is the average cost of the funds used to finance an investment opportunity. 
    The formula is as follows:
    WACC = (E / V) * Re + (D / V) * Rd * (1 - Tc)
    E = Market value of equity
    D = Market value of debt
    V = Total market value of the company (E + D)
    Re = Cost of equity refer to discount formula above
    Rd = Cost of debt, fee of loan, ie lown 100 at %1 and pay 1 dollar per year, 1 dollar is cost of the loan / debt
    tc = tax rate

    Tc = Corporate tax rate

    WACC = (Ke * We) + (Kd * (1 - Tc) * Wd)

    Let's assume the following values for XYZ Corp.:

    Cost of Equity (Ke): 10%
    Cost of Debt (Kd): 5%
    Equity Weight (We): 70%
    Debt Weight (Wd): 30%
    Corporate Tax Rate (Tc): 25%

    market value of debt, short + long term debt from balance ie total debt

    e/v = equity / total market value = equity weight
    d/v = debt / total market value = equity weight
    "While the market value of debt should be used,
    the book value of debt shown on the balance sheet is usually fairly close to the market value
    (and can be used as a proxy should the market value of debt not be available)."
    ex ke = 12,4058


    :param ticker: 
    :param key: 
    :return: 
    """
    Ticker = ticker
    Key = key
    Re = discount_rate / 100
    Rd = 0.0423 #vad e dehär?
    list = fa.balance_sheet_statement(Ticker, Key) #"totaldebt"
    marketcap = fa.quote(Ticker,key)
    d = list.iloc[48][0]
    e = marketcap.iloc[9][0]
    tc = 0.21
    v = e+d
    wacc = ((e / v) * Re) + (((d/v) * Rd) * (1 - tc))
    # test nvda wacc = 0.09456624650893691
    # valueinvest.io wacc = 9,5%, finbox wacc = 11,8, gurufocus wacc = 16,3, fmp wacc = 11,92
    #ej klar fixa cost of debt
    return wacc


