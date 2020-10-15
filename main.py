import tweepy

#override tweepy.StreamListener to add logic to on_status
class HIVStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)

def get_creds():
    creds = dict()

    with open('creds.txt', 'r') as f:
        for line in f:
            line = line.split("=")
            creds[line[0]] = line[1].strip()

    return creds


def query_builder():
    meds = []
    with open('meds.txt') as f:
        for line in f:
            meds.append(line.strip())
    print(meds)


def main():
    creds = get_creds()
    auth = tweepy.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
    auth.set_access_token(creds['access_token'], creds['access_token_secret'])
    api = tweepy.API(auth)

    listener = HIVStreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=listener)

    stream.filter(track=['bolsonaro'])


if __name__ == '__main__':
    main()
    # print(query_builder())
    # print(get_creds())
