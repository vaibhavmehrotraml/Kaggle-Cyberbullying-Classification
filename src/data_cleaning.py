import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../data/cyberbullying_tweets.csv')
    df = df[~df.tweet_text.isna()]
    df = df[~df.cyberbullying_type.isna()]
    df = df[df.tweet_text.str.len() < 300]
    df['tweet_text'] = df.tweet_text.apply(
        lambda x: ' '.join(word for word in x.split(' ') if not word.startswith('@')))
    df = df[df.cyberbullying_type != 'other_cyberbullying']
    df = df[~df.tweet_text.duplicated()]
    df = df[~(df.tweet_text == '')]
    df.to_csv('../data/cyberbullying_tweets_clean.csv', index=False)
