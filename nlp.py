import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def getFeatures(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


def dataPreprocessing(data):
    features = getFeatures(data)
    train_data = []

    for feature in features:
        data_single_feature = data[['title', feature]]
        test_df = data_single_feature.dropna()  # 1st iteration, drop all NaNs
        numpy_array = test_df.as_matrix()
        X = numpy_array[:, 0]  # words
        Y = numpy_array[:, 1]  # value of OS
        Y = Y.astype(int)  # need to cast to int for later use
        train_data.append((X, Y))

    return train_data


def train(train_data):
    text_clf_list = []
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    for data in train_data:
        X = data[0]
        Y = data[1]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.4, random_state=42)
        text_clf = text_clf.fit(X_train, Y_train)
        text_clf_list.append(text_clf)

    return text_clf_list


def test(text_clf_list, val_data_csv):
    val_data = pd.read_csv(val_data_csv, encoding='utf8')
    X_test = val_data['title']
    predicted_list = []

    for text_clf in text_clf_list:
        predicted = text_clf.predict(X_test)
        predicted_list.append(predicted)

    return predicted_list


if __name__ == "__main__":
    # Read the data
    mobile_data_train_csv = 'data/mobile_data_info_train_competition.csv'
    fashion_data_train_csv = 'data/fashion_data_info_train_competition.csv'
    beauty_data_train_csv = 'data/beauty_data_info_train_competition.csv'

    mobile_data = pd.read_csv(mobile_data_train_csv, encoding='utf8')
    fashion_data = pd.read_csv(fashion_data_train_csv, encoding='utf8')
    beauty_data = pd.read_csv(beauty_data_train_csv, encoding='utf8')

    # Data preprocessing (mobile)
    train_data = dataPreprocessing(mobile_data)

    # Train a model
    text_clf_list = train(train_data)

    # Test the model
    val_data_csv = 'data/mobile_data_info_val_competition.csv'
    predicted_list = test(text_clf_list, val_data_csv)
    print(predicted_list) # Note: This is only top-1 prediction. 

    #TODO: Get top-2 prediction

    #TODO: Create submission
