import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from features import category_json, category_feature_columns

#########################################
#   Sources:
#   https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
#   https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#########################################


def getFeatures(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


def df_class_to_text(df, category):
    """This function convert the entire numeric dataframe into text dataframe"""

    map_json = category_json[category]
    column_map = {}
    for column in category_feature_columns[category]:
        column_map[column] = {v: k for k, v in map_json[column].items()}
        df.loc[:, column] = df[column].map(column_map[column])

    return df


def dataPreprocessing(data, category):
    data_text = df_class_to_text(data, category)
    features = getFeatures(data)
    train_data_list = []

    for feature in features:
        data_single_feature = data_text[['title', feature]]
        test_df = data_single_feature.dropna()  # 1st iteration, drop all NaNs
        numpy_array = test_df.as_matrix()
        X = numpy_array[:, 0]
        Y = numpy_array[:, 1]
        train_data_list.append((X, Y))

    return train_data_list


def train(train_data_list):

    text_clf_trained_list = []

    for train_data in train_data_list:
        text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                             ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-3, n_iter=5, random_state=42)),
                             ])
        X = train_data[0]
        Y = train_data[1]
        text_clf_trained = text_clf.fit(X, Y)
        text_clf_trained_list.append(text_clf_trained)

    return text_clf_trained_list


def test(text_clf_list, val_data_csv):
    predicted_list = []
    val_data = pd.read_csv(val_data_csv, encoding='utf8')

    for text_clf in text_clf_list:
        X_test = val_data['title']
        predicted = text_clf.predict(X_test)
        predicted_list.append(predicted)

    return predicted_list


def savePredictionResults(predicted_list, dataset_type):
    dataset = 'data/' + dataset_type + '_data_info_val_competition.csv'
    validation_data = pd.read_csv(dataset, encoding='utf8')

    if (dataset_type == "mobile"):
        data = mobile_data
    elif (dataset_type == "fashion"):
        data = fashion_data
    else:
        data = beauty_data

    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    for i in range(len(getFeatures(data))):
        prediction = [entry for entry in predicted_list[i]]
        feature = getFeatures(data)[i]
        validation_data.insert(3+i, feature, prediction)

    prediction_csv = dataset_type + '_data_info_val_prediction_competition.csv'
    validation_data.to_csv("predictions/" + prediction_csv, index=False)
    print("Prediction results saved to " + prediction_csv + "\n")


######################################################################

if __name__ == "__main__":
    # Read the data
    mobile_data = pd.read_csv(
        'data/mobile_data_info_train_competition.csv', encoding='utf8')
    fashion_data = pd.read_csv(
        'data/fashion_data_info_train_competition.csv', encoding='utf8')
    beauty_data = pd.read_csv(
        'data/beauty_data_info_train_competition.csv', encoding='utf8')

    data_list = [mobile_data, fashion_data, beauty_data]

    for i in range(len(data_list)):
        data_type = ""
        if i == 0:
            data_type = "mobile"
        elif i == 1:
            data_type = "fashion"
        else:
            data_type = "beauty"

        # Data preprocessing
        train_data = dataPreprocessing(data_list[i], data_type)

        # Train a model
        text_clf_list = train(train_data)

        # Test the model
        val_data_csv = 'data/' + data_type + '_data_info_val_competition.csv'
        predicted_list = test(text_clf_list, val_data_csv)

        # Create CSV files for prediction results
        savePredictionResults(predicted_list, data_type)

    print("\nAll prediction results saved.")
