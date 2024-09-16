import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import joblib
from sklearn.metrics import accuracy_score

def num_char_split(df):
    num = df.select_dtypes(include = 'number')
    char = df.select_dtypes(include = 'object')
    print('Dataframe splitted')
    return num,char

def zero_threshold(num):
    selector = VarianceThreshold(threshold=0)  # threshold=0 means removing all zero-variance features
    num_reduced = selector.fit_transform(num)
    num = pd.DataFrame(num_reduced, columns=num.columns[selector.get_support(indices=True)])
    print("Variables with zero threshold dropped")
    return num

def concat(num,char):
    df = pd.concat([num, char], axis = 1)
    print("Concatinating done")
    return df

def drop_variables(df):
    df.drop(['split', 'original','id', 'related', 'genre'], axis = 1, inplace = True)
    print("Varibales dropped")
    return df

def drop_duplicate(df):
    df.drop_duplicates(inplace = True)
    print("Duplicates dropped")
    return df

def process(df):
    df['message'].replace("[^a-zA-Z]"," ",regex=True, inplace=True)
    df['message'] = df['message'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    print("Processing done")
    return df

def split_df(df):
    X = df['message']
    y = df[['request', 'aid_related', 'medical_help', 'medical_products',
           'search_and_rescue', 'security', 'military', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather']]
    print("Splitting done")
    return X,y

def vectorize(X):
    vectorizer = joblib.load('vectorizer.pkl')
    X = vectorizer.transform(X)
    print(X.shape)
    print("Vectorizer done")
    return X

def predict(X,y):
    print("Making prediction")
    model = joblib.load('multi_output_random_forest_model.pkl')
    y_pred = model.predict(X)
    y_pred_df = pd.DataFrame(y_pred, columns=y.columns)
    print("prediction done")
    return y_pred_df

def scores(y, y_pred):
    accuracy_scores = {}
    for column in y.columns:
        accuracy_scores[column] = accuracy_score(y[column], y_pred_df[column])

    accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Target Variable', 'Accuracy'])
    average_accuracy = accuracy_df['Accuracy'].mean()
    return accuracy_df, average_accuracy


if __name__ == '__main__':

    df = pd.read_csv('Data/disaster_response_messages_test.csv')
    num, char = num_char_split(df)
    num = zero_threshold(num)
    df = concat(num,char)
    df = drop_variables(df)
    df = drop_duplicate(df)
    df = process(df)
    X,y = split_df(df)
    X = vectorize(X)
    y_pred_df = predict(X,y)
    accuracy_df, average_accuracy = scores(y, y_pred_df)
    print(accuracy_df)
    print(average_accuracy)







   


    


