import pandas as pd


def getFeatures(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)

def createSubmissionDataframe(predictions, features):
    columns = ['id','tagging']
    rows = []

    for index, data in predictions.iterrows():
        for feature in features:
            id_label = str(data['itemid']) + "_" + feature
            row = [id_label, data[feature][1] + " " + data[feature][4]]
            rows.append(row)

    
    submission_df = pd.DataFrame(rows, columns=columns) 
    return submission_df


######################################################################

if __name__ == "__main__":
    # Read the saved prediction files 
    mobile_data_predictions = pd.read_csv('predictions/mobile_data_info_val_prediction_competition.csv', encoding='utf8')
    fashion_data_predictions = pd.read_csv('predictions/fashion_data_info_val_prediction_competition.csv', encoding='utf8')
    beauty_data_predictions = pd.read_csv('predictions/beauty_data_info_val_prediction_competition.csv', encoding='utf8')

    # Create the individual submission dataframes
    mobile_submission_df = createSubmissionDataframe(mobile_data_predictions, getFeatures(mobile_data_predictions))
    fashion_submission_df = createSubmissionDataframe(fashion_data_predictions, getFeatures(fashion_data_predictions))
    beauty_submission_df = createSubmissionDataframe(beauty_data_predictions, getFeatures(beauty_data_predictions))

    # Combine the submission dataframes into one
    submission_df = pd.concat([mobile_submission_df, fashion_submission_df, beauty_submission_df])
    submission_df.to_csv("predictions/data_info_val_submission.csv", index=False)
    print("Submission file created")

    
