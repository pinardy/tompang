import pandas as pd
from features import category_json, category_feature_columns


def getFeatures(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


def createSubmissionDataframe(predictions, features):
    columns = ['id', 'tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            row = [id_label, data[feature]]
            rows.append(row)

    submission_df = pd.DataFrame(rows, columns=columns)
    return submission_df


def df_text_to_class(df, category):
    """This function convert the entire numeric dataframe from text into class dataframe"""

    map_json = category_json[category]
    column_map = {}
    for column in category_feature_columns[category]:
        column_map[column] = {k: v for k, v in map_json[column].items()}
        df.loc[:, column] = df[column].map(column_map[column])

    return df


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
    mobile_submission_df = createSubmissionDataframe(
        mobile_data_predictions, getFeatures(mobile_data_predictions_classes))
    fashion_submission_df = createSubmissionDataframe(
        fashion_data_predictions, getFeatures(fashion_data_predictions_classes))
    beauty_submission_df = createSubmissionDataframe(
        beauty_data_predictions, getFeatures(beauty_data_predictions_classes))

    # Combine the submission dataframes into one
    submission_df = pd.concat(
        [mobile_submission_df, fashion_submission_df, beauty_submission_df])
    submission_df.to_csv(
        "predictions/data_info_val_submission.csv", index=False)
    print("Submission file created")
