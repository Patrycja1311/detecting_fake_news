import numpy as np
from download_data import data_read
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = data_read()


def check_null_title():
    cnt = df['title'].isnull()
    return cnt


def check_null_text():
    cntxt = df['text'].isnull()
    return cntxt


def check_null_label():
    cnl = df['label'].isnull()
    return cnl


def list_columns():
    list_colm = list(df.columns)
    return list_colm


def clean_column():
    clean_df = df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # df_clean = df.head()
    # return print(df_clean)
    return clean_df


def predict_model():
    x = np.array(df['title'])
    y = np.array(df['label'])
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    model = MultinomialNB().fit(xtrain, ytrain)

    headline_news = input('Enter a headline:')
    data_news = vectorizer.transform([headline_news]).toarray()
    return print(model.predict(data_news))


def count_fake():
    num_news = df.shape[0]
    count_fake_news = df['label'].str.count('FAKE').sum()
    pct_fake_news = round(count_fake_news/num_news, 2)
    return print(f'Total amount of news {num_news} pieces. Amount of fake news {count_fake_news} pieces. Percentage of fake news {pct_fake_news}%')
