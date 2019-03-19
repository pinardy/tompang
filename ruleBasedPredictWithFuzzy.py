import random
import pandas as pd
from pandas.io.json import json_normalize
from fuzzywuzzy import process, fuzz


android = ['samsung', 'xiaomi', 'oppo', 'google', 'android']
iOS = ['iphone', 'apple']
nokia = ['nokia']
blackberry = ['blackberry']

# todo complete all brand from json
allBrand = android + iOS + nokia + blackberry

mobile_features = ['Memory RAM', 'Network Connections', 'Storage Capacity', 'Color Family', 'Phone Model', 'Camera',
                   'Phone Screen Size']
beauty_features = ['Benefits', 'Brand', 'Colour_group', 'Product_texture', 'Skin_type']
fashion_features = ['Pattern', 'Collar Type', 'Fashion Trend', 'Clothing Material', 'Sleeves']


def get_features(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


def create_mobile_df(mobile_profile, labels, predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            title = str(data['title'])

            prediction1 = data[feature][1]
            prediction2 = data[feature][4]

            title = title.replace('biru', 'blue')
            title = title.replace('emas', 'gold')
            title = title.replace('coklat', 'brown')
            title = title.replace('cokelat', 'brown')
            title = title.replace('merah', 'red')
            title = title.replace('kuning', "yellow")
            title = title.replace('bening', "neutral")
            title = title.replace('polos', "neutral")
            title = title.replace('rosegold', "rose gold")
            title = title.replace('perak', "silver")
            title = title.replace('merah muda', "pink")
            title = title.replace('abu', "gray")
            title = title.replace('loreng', "army green")
            title = title.replace('army', "army green")
            title = title.replace('ungu', "purple")
            title = title.replace('grey', "light grey")
            title = title.replace('hitam', "black")
            title = title.replace('oranye', "orange")
            title = title.replace('jeruk', "orange")
            title = title.replace('hijau', "green")
            title = title.replace('putih', "white")
            title = title.replace('merah', 'red')

            if feature == 'Operating System':
                brands = process.extractOne(title, allBrand, fuzz.utils.full_process, fuzz.token_set_ratio)
                # print("brands", brands)
                brand = brands[0]

                if brand in android:
                    prediction1 = mobile_profile['Operating System']['android']
                    prediction2 = mobile_profile['Operating System']['samsung os']
                elif brand in iOS:
                    prediction1 = mobile_profile['Operating System']['ios']
                    prediction2 = mobile_profile['Operating System']['android']
                elif brand in nokia:
                    prediction1 = mobile_profile['Operating System']['android']
                    prediction2 = mobile_profile['Operating System']['windows']
                elif brand in blackberry:
                    prediction1 = mobile_profile['Operating System']['blackberry os']
                else:
                    prediction2 = prediction1
                    prediction1 = mobile_profile['Operating System']['android']

            if feature == 'Brand':
                brand = process.extract(title, labels['Brand'], fuzz.utils.full_process, fuzz.token_set_ratio,
                                        limit=2)
                # print("brand", brand)
                prediction1 = mobile_profile['Brand'][brand[0][0]]
                prediction2 = mobile_profile['Brand'][brand[1][0]]

                if any(keyword in title for keyword in iOS):
                    prediction1 = mobile_profile['Brand']['apple']
                    # print('Brand', prediction1)

            for mobile_feature in mobile_features:
                if feature == mobile_feature:
                    most_similar = process.extract(title, labels[feature], fuzz.utils.full_process,
                                                   fuzz.token_set_ratio, limit=2)
                    # print("similar", most_similar)
                    prediction1 = mobile_profile[mobile_feature][most_similar[0][0]]
                    prediction2 = mobile_profile[mobile_feature][most_similar[1][0]]
                    break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def create_fashion_submission_df(profile, labels, predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            title = str(data['title'])

            prediction1 = data[feature][1]
            prediction2 = data[feature][4]

            for fashion_feature in fashion_features:
                if feature == fashion_feature:
                    most_similar = process.extract(title, labels[feature], fuzz.utils.full_process,
                                                   fuzz.token_set_ratio, limit=2)
                    prediction1 = profile[feature][str(most_similar[0][0])]
                    prediction2 = profile[feature][most_similar[1][0]]
                    break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def create_beauty_submission_df(profile, labels, predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            title = str(data['title'])

            title = title.replace('blue', 'biru')
            title = title.replace('gold', 'emas')
            title = title.replace('brown', 'cokelat')
            title = title.replace('red', 'merah')
            title = title.replace("yellow", 'kuning')
            title = title.replace("neutral", 'netral')
            title = title.replace("rose", 'mawar')
            title = title.replace("rose gold", 'emas rose')
            title = title.replace("rosegold", 'emas rose')
            title = title.replace("silver", 'perak')
            title = title.replace("pink", 'merah muda')
            title = title.replace("gray", 'abu')
            title = title.replace("purple", 'ungu')
            title = title.replace("black", 'hitam')
            title = title.replace("green", 'hijau')
            title = title.replace('white', 'putih')
            title = title.replace('red', 'merah')
            title = title.replace('multicolor', 'multiwarna')
            title = title.replace('coral', 'warna koral')
            title = title.replace('cherry', 'ceri')
            title = title.replace('orange', 'jeruk')

            prediction1 = data[feature][1]
            prediction2 = data[feature][4]

            for beauty_feature in beauty_features:
                if feature == beauty_feature:
                    most_similar = process.extract(title, labels[feature], fuzz.utils.full_process,
                                                   fuzz.token_set_ratio, limit=2)
                    # print("similar", most_similar)
                    prediction1 = profile[feature][most_similar[0][0]]
                    prediction2 = profile[feature][most_similar[1][0]]
                    break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    mobile_data_predictions = pd.read_csv('predictions/mobile_data_info_val_prediction_competition.csv',
                                          encoding='utf8')
    fashion_data_predictions = pd.read_csv('predictions/fashion_data_info_val_prediction_competition.csv',
                                           encoding='utf8')
    beauty_data_predictions = pd.read_csv('predictions/beauty_data_info_val_prediction_competition.csv',
                                          encoding='utf8')

    mobile_json = pd.read_json('data/mobile_profile_train.json', typ='series')
    mobile_normalized = json_normalize(mobile_json)[0]
    mobile_labels = {}

    for category in list(mobile_normalized.keys()):
        mobile_labels[str(category)] = list(mobile_normalized[str(category)].keys())

    mobile_submission_df = create_mobile_df(
        mobile_json,
        mobile_labels,
        mobile_data_predictions,
        get_features(mobile_data_predictions)
    )

    fashion_json = pd.read_json('data/fashion_profile_train.json', typ='series')
    fashion_normalized = json_normalize(fashion_json)[0]
    fashion_labels = {}

    for category in list(fashion_normalized.keys()):
        fashion_labels[str(category)] = list(fashion_normalized[str(category)].keys())

    # print('fashion_labels', fashion_labels)

    fashion_submission_df = create_fashion_submission_df(
        fashion_json,
        fashion_labels,
        fashion_data_predictions,
        get_features(fashion_data_predictions)
    )

    beauty_json = pd.read_json('data/beauty_profile_train.json', typ='series')
    beauty_normalized = json_normalize(beauty_json)[0]
    beauty_labels = {}

    for category in list(beauty_normalized.keys()):
        beauty_labels[str(category)] = list(beauty_normalized[str(category)].keys())

    beauty_submission_df = create_beauty_submission_df(
        beauty_json,
        beauty_labels,
        beauty_data_predictions,
        get_features(beauty_data_predictions)
    )

    # Combine the submission dataframes into one
    submission_df = pd.concat([mobile_submission_df, fashion_submission_df, beauty_submission_df])
    submission_df.to_csv("predictions/data_info_val_submission.csv", index=False)
    # mobile_submission_df.to_csv("predictions/data_info_val_submission.csv", index=False)
    print("Submission file created")
