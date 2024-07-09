# import libraries
import sys
import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine

from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

#import nltk

#nltk.download('wordnet')

def load_data(database_filepath):
    """
    Import dataseet from SQLite database.

    Parameters
    ----------
        database_filepath: str

    Returns:
        pandas.Series, list : X, Y, category_names.
    """

    # load data from sqlite Database
    engine = create_engine(f'sqlite:///{database_filepath}')

    sql_script = "SELECT * FROM categories;"

    df = pd.read_sql(sql_script, engine)

    print(df.head())
    
    # features and target
    X = df.message.values
    Y = df.genre.values    

    category_names = list(df.columns)[3:]
    
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize text (normalize, lemmatize, tokenize)

    Parameters
    ----------
        text:str to be tokenized.

    Return:
        list:text tokens.     
    """
    # Eliminate punctations
    text =  ''.join([c for c in text if c not in punctuation])
    
    #tokenize text
    tokens = word_tokenize(text,preserve_line=True)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token_x in tokens:
        clean_tok = lemmatizer.lemmatize(token_x).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Build and return a machine learning model.
     Input:
       None
    Return:
        Model: Machine learning model.
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', RandomForestClassifier())
    ])
    # grid search parameters
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
    #create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate ML Model performance on test data

    Args:
        model: Machine learning model.
        X_test: Test set variables.
        Y_test: Target Variable for test set.
        category_names: List of categories.

    Return:
        None
    """
    Y_pred = model.predict(X_test)
    # confusion_mat = confusion_matrix(Y_test, Y_pred, labels=category_names)
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    accuracy = (Y_pred == Y_test).mean()
    
    print("Categories: \n", category_names)
    print("Model Accuracy: ", accuracy)
    print("Confusion Matrix: \n", confusion_mat)
    print("\n Top Important Features: ", model.best_params_)


def save_model(model, model_filepath):
    """
    Save model to as pickle file.

    Input:
        model: Machine learning model.
        model_filepath (str): Location/Filepath to save the model.

    Return:
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Main function to train and evaluate a machine learning model.

    Reads command line arguments, loads data, trains the model, evaluates it, and saves the       model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(len(X_train), len(X_test), len(Y_train), len(Y_test), len(category_names))
        
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