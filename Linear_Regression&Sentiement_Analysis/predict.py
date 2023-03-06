# Import required packages 
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import tweepy

# Twitter API credentials
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# authenticate Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# download historical price data for Bitcoin
btc = yf.download("BTC-USD", period="3y")

# save the data as a CSV file
btc.to_csv("historical_data.csv", index=False)

# load the data from the CSV file
historical_data = pd.read_csv("historical_data.csv")

# calculate the average price over 30, 60, and 90 day periods
historical_data["30_day_avg"] = historical_data["Close"].rolling(window=30).mean()
historical_data["60_day_avg"] = historical_data["Close"].rolling(window=60).mean()
historical_data["90_day_avg"] = historical_data["Close"].rolling(window=90).mean()

# retrieve top 100 tweets related to Bitcoin
tweets = api.search(q="Bitcoin", count=100)

# calculate sentiment score of each tweet
sentiment_scores = []
for tweet in tweets:
    sentiment_scores.append(TextBlob(tweet.text).sentiment.polarity)
    
# calculate the average sentiment score
average_sentiment = np.mean(sentiment_scores)

# create a new dataframe for the input features
X = historical_data[["30_day_avg", "60_day_avg", "90_day_avg"]]
X["sentiment"] = average_sentiment

# create a new dataframe for the target variable
y = historical_data["Close"].shift(-1)

# split the data into training and testing sets
X_train = X[:-30]
X_test = X[-30:]
y_train = y[:-30]
y_test = y[-30:]

# fit the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# make a prediction on the most recent day's data
last_row = historical_data.tail(1)
last_row_features = last_row[["30_day_avg", "60_day_avg", "90_day_avg"]]
last_row_sentiment = average_sentiment
last_row_features["sentiment"] = last_row_sentiment
predicted_price = lr_model.predict(last_row_features)[0]

# make predictions on the test set
y_pred = lr_model.predict(X_test)

# calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)

# print the predicted price and RMSE
print("Predicted price for the next day: ", predicted_price)
print("Root Mean Squared Error: ", rmse)
