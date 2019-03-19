import pandas as pd
from features import category_json, category_feature_columns


android = ['samsung', 'xiaomi', 'oppo', 'google', 'android', ]
iOS = ['iphone', 'apple']
nokia = ['nokia']
blackberry = ['blackberry']

mobile_features = ['Memory RAM', 'Network Connections', 'Storage Capacity', 'Phone Model', 'Camera',
                   'Phone Screen Size']
beauty_features = ['Benefits', 'Brand', 'Product_texture', 'Skin_type']
fashion_features = ['Pattern', 'Collar Type', 'Fashion Trend', 'Clothing Material', 'Sleeves']


def get_features(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


def df_text_to_class(df, category):
    """This function convert the entire numeric dataframe from text into class dataframe"""

    map_json = category_json[category]
    column_map = {}
    for column in category_feature_columns[category]:
        column_map[column] = {k: v for k, v in map_json[column].items()}
        df.loc[:, column] = df[column].map(column_map[column])

    return df


def create_mobile_df(mobile_profile, predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            title = str(data['title'])

            prediction1 = data[feature]
            prediction2 = data[feature]

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
                    prediction2 = prediction1
                    prediction1 = mobile_profile['Operating System']['android']

            if feature == 'Brand':
                for brand in mobile_profile['Brand']:
                    if brand in title:
                        prediction1 = mobile_profile['Brand'][brand]
                        # print('Brand', brand, prediction1)
                        break
                if any(keyword in title for keyword in iOS):
                    prediction1 = mobile_profile['Brand']['apple']
                    # print('Brand', 'apple', prediction1)

            if feature == 'Color Family':
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
                for color in mobile_profile['Color Family']:
                    if color in title:
                        prediction1 = mobile_profile['Color Family'][color]
                        # print('color', prediction1)
                        break

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

            prediction1 = data[feature]
            prediction2 = data[feature]

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

            prediction1 = data[feature]
            prediction2 = data[feature]

            for beauty_feature in beauty_features:
                if feature == beauty_feature:
                    for keyword in profile[beauty_feature]:
                        if keyword in title:
                            prediction1 = profile[beauty_feature][keyword]
                            break

            if feature == 'Colour_group':
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
                # print('title', title)

                for color in profile['Colour_group']:
                    if color in title:
                        prediction1 = profile['Colour_group'][color]
                        # print('color beauty', color)
                        break

            row = [id_label, str(prediction1) + ' ' + str(prediction2)]
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


######################################################################

if __name__ == "__main__":
    # Read the saved prediction files
    mobile_data_predictions = pd.read_csv(
        'predictions/mobile_data_info_val_prediction_competition.csv', encoding='utf8')
    fashion_data_predictions = pd.read_csv(
        'predictions/fashion_data_info_val_prediction_competition.csv', encoding='utf8')
    beauty_data_predictions = pd.read_csv(
        'predictions/beauty_data_info_val_prediction_competition.csv', encoding='utf8')

    # Convert from text to classes (numbers)
    mobile_data_predictions_classes = df_text_to_class(
        mobile_data_predictions, "mobile")
    fashion_data_predictions_classes = df_text_to_class(
        fashion_data_predictions, "fashion")
    beauty_data_predictions_classes = df_text_to_class(
        beauty_data_predictions, "beauty")

    # Create the individual submission dataframes
    mobile_submission_df = create_mobile_df(
        pd.read_json('data/mobile_profile_train.json', typ='series'),
        mobile_data_predictions,
        get_features(mobile_data_predictions_classes)
    )

    fashion_submission_df = create_fashion_submission_df(
        pd.read_json('data/fashion_profile_train.json', typ='series'),
        fashion_data_predictions,
        get_features(fashion_data_predictions_classes)
    )

    beauty_submission_df = create_beauty_submission_df(
        pd.read_json('data/beauty_profile_train.json', typ='series'),
        beauty_data_predictions,
        get_features(beauty_data_predictions_classes)
    )

    # Combine the submission dataframes into one
    submission_df = pd.concat(
        [mobile_submission_df, fashion_submission_df, beauty_submission_df])
    submission_df.to_csv(
        "predictions/data_info_val_submission.csv", index=False)
    print("Submission file created")
