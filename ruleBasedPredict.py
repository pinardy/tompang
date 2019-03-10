import random

import pandas as pd

android = ['samsung', 'xiaomi', 'oppo', 'google', 'android', ]
iOS = ['iphone', 'apple']
nokia = ['nokia']
blackberry = ['blackberry']

mobile_features = ['Memory RAM', 'Network Connections', 'Storage Capacity', 'Color Family', 'Phone Model', 'Camera',
                   'Phone Screen Size']
beauty_features = ['Benefits', 'Brand', 'Colour_group', 'Product_texture', 'Skin_type']
fashion_features = ['Pattern', 'Collar Type', 'Fashion Trend', 'Clothing Material', 'Sleeves']


def get_features(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


def create_mobile_df(mobile_profile, predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            title = str(data['title'])

            prediction1 = data[feature][1]
            prediction2 = data[feature][4]

            if feature == 'Operating System':
                if any(os in title for os in android):
                    prediction1 = mobile_profile['Operating System']['android']
                    prediction2 = mobile_profile['Operating System']['samsung os']
                elif any(os in title for os in iOS):
                    prediction1 = mobile_profile['Operating System']['ios']
                    prediction2 = mobile_profile['Operating System']['android']
                elif any(os in title for os in nokia):
                    prediction1 = mobile_profile['Operating System']['android']
                    prediction2 = mobile_profile['Operating System']['windows']
                elif any(os in title for os in blackberry):
                    prediction1 = mobile_profile['Operating System']['blackberry os']
                else:
                    prediction2 = mobile_profile['Operating System']['android']

            if feature == 'Brand':
                for brand in mobile_profile['Brand']:
                    if brand in title:
                        prediction1 = mobile_profile['Brand'][brand]
                        break
                if any(keyword in title for keyword in iOS):
                    prediction1 = mobile_profile['Brand']['apple']
                prediction2 = random.randrange(len(mobile_profile['Brand']))

            for mobile_feature in mobile_features:
                if feature == mobile_feature:
                    for keyword in mobile_profile[mobile_feature]:
                        if keyword in title:
                            prediction1 = mobile_profile[mobile_feature][keyword]
                            break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def create_fashion_submission_df(profile, predictions, features):
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
                    for keyword in profile[fashion_feature]:
                        if keyword in title:
                            prediction1 = profile[fashion_feature][keyword]
                            break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def create_beauty_submission_df(profile, predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            title = str(data['title'])

            prediction1 = data[feature][1]
            prediction2 = data[feature][4]

            for beauty_feature in beauty_features:
                if feature == beauty_feature:
                    for keyword in profile[beauty_feature]:
                        if keyword in title:
                            prediction1 = profile[beauty_feature][keyword]
                            break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    # Read the saved prediction files
    mobile_data_predictions = pd.read_csv('predictions/mobile_data_info_val_prediction_competition.csv',
                                          encoding='utf8')
    fashion_data_predictions = pd.read_csv('predictions/fashion_data_info_val_prediction_competition.csv',
                                           encoding='utf8')
    beauty_data_predictions = pd.read_csv('predictions/beauty_data_info_val_prediction_competition.csv',
                                          encoding='utf8')

    # Create the individual submission dataframes
    mobile_submission_df = create_mobile_df(
        pd.read_json('data/mobile_profile_train.json', typ='series'),
        mobile_data_predictions,
        get_features(mobile_data_predictions)
    )

    fashion_submission_df = create_fashion_submission_df(
        pd.read_json('data/fashion_profile_train.json', typ='series'),
        fashion_data_predictions,
        get_features(fashion_data_predictions)
    )

    beauty_submission_df = create_beauty_submission_df(
        pd.read_json('data/beauty_profile_train.json', typ='series'),
        beauty_data_predictions,
        get_features(beauty_data_predictions)
    )

    # Combine the submission dataframes into one
    submission_df = pd.concat([mobile_submission_df, fashion_submission_df, beauty_submission_df])
    submission_df.to_csv("predictions/data_info_val_submission.csv", index=False)
    print("Submission file created")
