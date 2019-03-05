import pandas as pd


def getFeatures(data):
    features = data.drop(columns=['itemid', 'title', 'image_path'])
    return list(features)


######################################################################

if __name__ == "__main__":
    # Read the saved prediction files 
    mobile_data_predictions = pd.read_csv('predictions/mobile_data_info_val_prediction_competition.csv', encoding='utf8')
    fashion_data_predictions = pd.read_csv('predictions/fashion_data_info_val_prediction_competition.csv', encoding='utf8')
    beauty_data_predictions = pd.read_csv('predictions/beauty_data_info_val_prediction_competition.csv', encoding='utf8')

    #TODO: Create submission file with the saved prediction files

