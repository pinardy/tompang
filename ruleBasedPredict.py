import random

import pandas as pd

android = ['samsung', 'xiaomi', 'oppo', 'google', 'android', ]
iOS = ['iphone', 'apple']
nokia = ['nokia']
blackberry = ['blackberry']


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

            if feature == 'Memory RAM':
                for mem in mobile_profile['Memory RAM']:
                    if mem in title:
                        prediction1 = mobile_profile['Memory RAM'][mem]
                        break

            if feature == 'Network Connections':
                for net_con in mobile_profile['Network Connections']:
                    if net_con in title:
                        prediction1 = mobile_profile['Network Connections'][net_con]
                        break

            if feature == 'Storage Capacity':
                for keyword in mobile_profile['Storage Capacity']:
                    if keyword in title:
                        prediction1 = mobile_profile['Storage Capacity'][keyword]
                        break

            if feature == 'Color Family':
                for keyword in mobile_profile['Color Family']:
                    if keyword in title:
                        prediction1 = mobile_profile['Color Family'][keyword]
                        break

            if feature == 'Phone Model':
                for keyword in mobile_profile['Phone Model']:
                    if keyword in title:
                        prediction1 = mobile_profile['Phone Model'][keyword]
                        break

            if feature == 'Camera':
                for keyword in mobile_profile['Camera']:
                    if keyword in title:
                        prediction1 = mobile_profile['Camera'][keyword]
                        break

            if feature == 'Phone Screen Size':
                for keyword in mobile_profile['Phone Screen Size']:
                    if keyword in title:
                        prediction1 = mobile_profile['Phone Screen Size'][keyword]
                        break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    submission_df = pd.DataFrame(rows, columns=columns)
    return submission_df


def create_submission_df(predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            row = [id_label, data[feature][1] + " " + data[feature][4]]
            rows.append(row)

    submission_df = pd.DataFrame(rows, columns=columns)
    return submission_df


if __name__ == "__main__":
    # Read the saved prediction files
    mobile_data_predictions = pd.read_csv('predictions/mobile_data_info_val_prediction_competition.csv',
                                          encoding='utf8')
    fashion_data_predictions = pd.read_csv('predictions/fashion_data_info_val_prediction_competition.csv',
                                           encoding='utf8')
    beauty_data_predictions = pd.read_csv('predictions/beauty_data_info_val_prediction_competition.csv',
                                          encoding='utf8')

    mobile_profile = pd.read_json('data/mobile_profile_train.json', typ='series')

    # Create the individual submission dataframes
    mobile_submission_df = create_mobile_df(
        mobile_profile,
        mobile_data_predictions,
        get_features(mobile_data_predictions)
    )

    fashion_submission_df = create_submission_df(fashion_data_predictions, get_features(fashion_data_predictions))
    beauty_submission_df = create_submission_df(beauty_data_predictions, get_features(beauty_data_predictions))

    # Combine the submission dataframes into one
    submission_df = pd.concat([mobile_submission_df, fashion_submission_df, beauty_submission_df])
    submission_df.to_csv("predictions/data_info_val_submission.csv", index=False)
    print("Submission file created")
