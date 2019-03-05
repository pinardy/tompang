import pandas as pd
import numpy as np
import os 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#########################################
#   Sources:
#   https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
#   https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#########################################

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

    print("Training started...")
    for data in train_data:
        X = data[0]
        Y = data[1]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.4, random_state=42)
        text_clf = text_clf.fit(X_train, Y_train)
        text_clf_list.append(text_clf)
    
    print("Training complete!")
    return text_clf_list


def test(text_clf_list, val_data_csv):
    val_data = pd.read_csv(val_data_csv, encoding='utf8')
    X_test = val_data['title']
    predicted_list = []

    print("Making predictions...")
    for text_clf in text_clf_list:
        probs = text_clf.predict_proba(X_test)
        top_2 = np.argsort(probs, axis=1)[:, -2:]  
        predicted_list.append(top_2)
    print("Predictions complete!")
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
        prediction = [tuple(entry) for entry in predicted_list[i]]
        feature = getFeatures(data)[i]
        validation_data.insert(3+i, feature, prediction) 

    prediction_csv = dataset_type + '_data_info_val_prediction_competition.csv'
    validation_data.to_csv("predictions/" + prediction_csv, index=False)
    print("Prediction results saved to " + prediction_csv + "\n")


######################################################################

if __name__ == "__main__":
    # Read the data
    mobile_data = pd.read_csv('data/mobile_data_info_train_competition.csv', encoding='utf8')
    fashion_data = pd.read_csv('data/fashion_data_info_train_competition.csv', encoding='utf8')
    beauty_data = pd.read_csv('data/beauty_data_info_train_competition.csv', encoding='utf8')

    data_list = [mobile_data, fashion_data, beauty_data]

    for i in range(len(data_list)):
        if i == 0:
            data_type = "mobile"
        elif i == 1:
            data_type = "fashion"
        else:
            data_type = "beauty"

        # Data preprocessing 
        train_data = dataPreprocessing(data_list[i])

        # Train a model
        text_clf_list = train(train_data)

        # Test the model
        val_data_csv = 'data/' + data_type + '_data_info_val_competition.csv'
        predicted_list = test(text_clf_list, val_data_csv)

        # Create CSV files for prediction results
        savePredictionResults(predicted_list, data_type)  

    print("\nAll prediction results saved.")

    