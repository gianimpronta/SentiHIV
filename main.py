from textblob import TextBlob as tb
import tweepy
import numpy as np


def get_creds():
    creds = dict()

    with open('creds.txt', 'r') as f:
        creds['consumer_key'] = f.readline()
        creds['consumer_secret'] = f.readline()
        creds['access_token'] = f.readline()
        creds['access_token_secret'] = f.readline()
    return creds


def main():
    creds = get_creds()
    auth = tweepy.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
    auth.set_access_token(creds['access_token'], creds['access_token_secret'])
    api = tweepy.API(auth)

    # Variável que irá armazenar todos os Tweets com a palavra escolhida na função search da API
    public_tweets = api.search('Trump')

    tweets = []  # Lista vazia para armazenar scores
    for tweet in public_tweets:
        print(tweet.text)
        analysis = tb(tweet.text)
        polarity = analysis.sentiment.polarity
        tweets.append(polarity)
        print(polarity)

    print('MÉDIA DE SENTIMENTO: ' + str(np.mean(tweets)))


if __name__ == '__main__':
    main()
