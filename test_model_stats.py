from group_base_model import generate_features
from utility import load_instances, load_labels, load_timestamps, convert_to_classlabels, write_results
from sklearn.externals import joblib
import os


def test_model(*args):
    """
        param X_test: a feature matrix
        param model:
        return : predicted labels for test data to the file saved with model
        name
    """
    print(True)
    data_location = args[0]
    model_location = args[1]
    test_instances = load_instances(data_location)
    X_test = generate_features(test_instances, feature='touch')
    timestamps = load_timestamps(test_instances)
    for file in os.listdir(model_location):
        model = joblib.load(model_location + file)
        y_test = model.predict(X_test)
        file_name = file.strip('.pkl')
        classlabels = convert_to_classlabels(y_test)
        write_results(timestamps, classlabels, file_name + ".csv")


if __name__ == '__main__':
    model_loc = '../model_saver/'
    data_loc = '../data/test/'
    test_model(data_loc, model_loc)
