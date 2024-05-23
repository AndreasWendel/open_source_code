import pandas as pd
import requests
from polygon import RESTClient
from datetime import timedelta
import matplotlib.pyplot as plt


def graph(useddata, idnumber, titel, xlabel, size_x, size_y):
    plt.figure(idnumber, figsize=(size_x, size_y))
    plt.title(titel)
    plt.xlabel(xlabel)
    plt.plot(useddata)
    plt.legend()
    return


# datasets to req
groupName = "otcmarket"
datasetName = "regShoDaily"

# underlying
tkr = "SPY"

# build url
url = f'https://api.finra.org/data/group/{groupName}/name/{datasetName}'

# headers
headers = {
    'Content-TYpe': 'application/json',
    'Accept': 'application/json',
}
# create custom filter
customFilter = {
    "limit": 5000,
    "compareFilters": [
        {"compareType": "equal",
         "fieldName": "securitiesInformationProcessorSymbolIdentifier",
         "fieldValue": tkr}
    ]
}

"""customFilter = {
    "limit": 1000,
    "compareFilters": [
        {"compareType": "equal",
         "fieldName": "securitiesInformationProcessorSymbolIdentifier",
         "fieldValue": tkr}
    ],
    "dateRangeFilters": [ {

    "startDate" : tempstartdate,

    "endDate" : tempenddate,

    "fieldName" : "trade_time"

  } ]
}"""

# make POST request
request = requests.post(url, headers=headers, json=customFilter)

# create dataframe
data = pd.DataFrame.from_dict(request.json())
# data.info() ser alla typer av info, vill ha datum sortering
# format data , to_datetime converts argument from common date type/ yy,mm,dd to a datetime
data.tradeReportDate = pd.to_datetime(data.tradeReportDate)

# define agg function to apply, sum up shit
aggFunc = {'totalParQuantity': 'sum',
           'shortParQuantity': 'sum',
           'shortExemptParQuantity': 'sum'}

aggData = data.groupby(['tradeReportDate']).agg(aggFunc)
aggData.index.names = ['Date']
aggData.columns = ['volumeFINRA', 'shortVolumeFINRA', 'shortExemptVolumeFINRA']

# getting date for volume
startDate = aggData.index[-1] - timedelta(days=365)
endDate = aggData.index[-1] + timedelta(days=0)
# api key and client
#insert your own polygonkey
polygonapikey = ""
client = RESTClient(polygonapikey)  # api_key is used

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# enter date in yyyy-mm-dd format
tempstartdate = "start date"
tempenddate = "end date"
underlying = "SPY"
# get historical data
technicalData = client.get_aggs(
    underlying,
    1,
    "day",
    startDate,
    endDate,
    limit=10000
)
techdata = pd.DataFrame(technicalData)
print(type(techdata))
# add total colume column and making it to an int
techdata['Date'] = techdata['timestamp'].apply(lambda x: pd.to_datetime(x * 1000000))

aggData['totalVolume'] = techdata['volume'].values

aggData.info()

print()

# mathing vols and shit
# short vol (finra) over total vol(finra)
aggData['shortVolOverTVFINRA'] = aggData['shortVolumeFINRA'] / aggData['volumeFINRA']

# short vol (finra) over total Volume
aggData['shortVolOverTV'] = aggData['shortVolumeFINRA'] / aggData['totalVolume']

# short exemp vol (finra) over short vol (finra)
aggData['shortExemptVolOverSVFINRA'] = aggData['shortExemptVolumeFINRA'] / aggData['shortVolumeFINRA']

# short exemp vol (finra) over short vol (finra)
aggData['shortExemptVolOverTVFINRA'] = aggData['shortExemptVolumeFINRA'] / aggData['volumeFINRA']

# short exempt vol (finra) over total Volume
aggData['ShortExemptVoloverTV'] = aggData['shortExemptVolumeFINRA'] / aggData['totalVolume']

# plt.plot(aggData['shortVolOverTVFINRA'])
'''plt.plot(aggData['shortVolOverTV'])
plt.plot(aggData['shortExemptVolOverSVFINRA'])
plt.plot(aggData['shortExemptVolOverTVFINRA'])
plt.plot(aggData['ShortExemptVoloverTV'])
plt.show()'''

graph(aggData['shortVolOverTVFINRA'], 1, 'shortvoloverTVFINRA', 'date', 7, 4)
graph(aggData['shortVolOverTV'], 2, 'shortVolOverTV', 'date', 7, 4)
graph(aggData['shortExemptVolOverSVFINRA'], 3, 'shortExemptVolOverSVFINRA', 'date', 7, 4)
graph(aggData['shortExemptVolOverTVFINRA'], 4, 'shortExemptVolOverTVFINRA', 'date', 7, 4)
graph(aggData['ShortExemptVoloverTV'], 5, 'ShortExemptVoloverTV', 'date', 7, 4)
graph(aggData['shortExemptVolumeFINRA'], 6, 'shortExemptVolumeFINRA', 'date', 7, 4)
graph(aggData['shortVolumeFINRA'], 7, 'shortVolumeFINRA', 'date', 7, 4)
plt.show()
