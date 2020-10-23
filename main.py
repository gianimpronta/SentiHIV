import json

import tweepy


class HIVStreamListener(tweepy.StreamListener):

    def on_connect(self):
        print('Listener conectado')

    def on_status(self, status):
        print(status.text)
        write_tweet(status)

    def on_error(self, status_code):
        print(f'Erro: {status_code}')
        return

    def on_timeout(self):
        print('Timeout')
        return

    def on_disconnect(self, notice):
        print(notice)
        print('Desconectado')
        return


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

    # meds = ' OR '.join(meds)
    return meds


def write_tweet(status):
    print(json.dumps(status._json))


def main():
    creds = get_creds()
    auth = tweepy.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
    auth.set_access_token(creds['access_token'], creds['access_token_secret'])
    api = tweepy.API(auth)

    listener = HIVStreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=listener)

    query = query_builder()
    stream.filter(track=query, languages=['pt'])
    # stream.filter(track=['Trump'])


if __name__ == '__main__':
    main()
    # print(query_builder())
    # print(get_creds())
