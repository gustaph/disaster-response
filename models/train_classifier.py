import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from pickle import dump

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath: str):
    '''
    load the dataset from database
    
    Args:
        - database_filepath: str -> database directory
    Returns:
        - X: np.array -> x values for modeling
        - Y: np.array -> y values for modeling
        - category_columns: list
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}') # access data from database
    df = pd.read_sql_table('disaster_response', engine) # get data from database

    category_columns = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns

    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

    return X, Y, category_columns


def tokenize(text: str):
    '''
    function that transforms raw text into clean tokens
    
    Args:
        - text: str -> text to be tokenized
    Returns:
        - token list
    '''
    
    # regex obtained from Udacity course classes
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    clean_text = re.sub(r'[^a-z-A-Z]|\W', ' ', text.lower()) # remove everything but characters
    clean_text = re.sub(url_regex, 'urlplaceholder', clean_text) # replace url's by `urlplaceholder`
    
    tokens = word_tokenize(clean_text) # tokenize text
    lemmatizer = WordNetLemmatizer() # instatiate lemmatizer
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip() # lemmatize word
        clean_token = lemmatizer.lemmatize(clean_token, pos='v') # lemmatize again operating by verbs
        
        if clean_token not in stopwords.words("english"): # remove stopwords
            clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    '''
    build the ML model

    Returns:
        - ideal model based on GridSearch

    > REFERENCES:
        * Understanding LightGBM Parameters (and How to Tune Them) [https://neptune.ai/blog/lightgbm-parameters-guide]
    '''

    # first set of parameters based on references
    PARAMS = { 
        'learning_rate': 0.2,
        'max_depth': 10,
        'num_leaves': 20,
        'feature_fraction': 0.6,
        'n_estimators': 150,
        'verbose': -1
    }
    
    # model pipeline
    pipeline = Pipeline([
        ('count', CountVectorizer(tokenizer=tokenize)), # bag of words based on `tokenize` function
        ('tfidf', TfidfTransformer()), # TF-IDF transformer
        ('clf', MultiOutputClassifier(estimator=LGBMClassifier(**PARAMS), n_jobs=-1)), 
    ])

    PARAMS = {
        'clf__estimator__subsample': (0.1, 0.2, 0.5),
        'clf__estimator__class_weight': (None, 'balanced'),
    }

    grid_search = GridSearchCV(pipeline, param_grid=PARAMS) # operating GridSearch to find best parameters
    return grid_search


def evaluate_model(model, X_test: np.array, Y_test: np.array, category_names: list):
    '''
    shows the overall ranking of each target column
    
    Args:
        - model: classification model
        - X_test: x values for test
        - Y_test: y values for test
        - category_names
    Returns: all column classification report
    '''
    
    y_pred = model.predict(X_test) # predict data
    
    for i, column in enumerate(category_names):
        print(f"({column}):")
        print(classification_report(Y_test[:,i], y_pred[:,i], zero_division=1))
    
    print("ACC:", (y_pred == Y_test).mean())
        
    return


def save_model(model, model_filepath: str):
    '''
    export the model in pickle format
    
    Args:
        - model: final model
        - model_filepath: str -> directory from where to save the model
    '''
    
    with open(model_filepath, 'wb') as file:
        dump(model, file)

    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
