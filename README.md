# STI Constituents Clustering
Clustering of STI constituents using k-means

### Cluster Anaylsis
Observing how the various constituents of the STI move together based on Q4/2019 and Q1/2020 data. Cluster analysis can be used to help build diversified trading portfolios, since stocks that exhibit high correlations usually fall into one basket. 

The intention here is to have 'baskets' of stocks that are diversified enough (__exhibiting minimal correlation from one another__) to protect the investor against systemic/market risks. Hence, I wanted to take a further look into the constituents of the STI to see if they could be further seperated into different 'baskets', instead of following the index in its entirety. 

### Getting the data
After obtaining the tickers of the 30 stocks within the STI, Q4/2019 and Q1/2020 data was obtained from Yahoo Finance throuhg the pandas.datareader module. For this project, only the Adjusted Closing prices of each stock was used for analysis.

```python
#Getting the historical data for the 30 stocks
stidf = pd.read_csv('sti.csv')
tickers = stidf['Symbol'].tolist()
names = stidf['Name'].tolist()
df = dr.DataReader(tickers,'yahoo',start = '2019-10-01', end='2020-04-30')['Adj Close']
```

Fortunately, our data has no empty values, so no data cleaning is required.

### Exploring the data
```python
#Replace tickers with company names
df.columns = stidf['Name'].tolist()

#Some exploratory analysis of the data
df.head()
df.describe()
```

A correlation matrix will also tell us how the closing prices of the stocks move in relation to one another, since the correlation coefficient is a measure of strength of that relationship. 

```python
#correlation matrix
corrMatrix = df.corr()
corrMatrix
```
![Sample of correlation matrix]
