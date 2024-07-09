#Import Libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load  message and categories files,merge them and return a new dataframe 

    Parameters:
        messages_filepath: (str) CSV file.
        categories_filepath : (str) CSV file.

    Return:
        Merged pandas DataFrame.
    """
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id')
        
    return df
    
    


def clean_data(df):
    """
    Clean DataFrame, 
        expanding the multiple categories into seperate columns, 
        extract categories values, 
        replace the previous categories with new columns
        removing duplicates

    Args:
        df:dataframe containing messages and categories.

    Returns:
        DataFrame: Cleaned dataframe.

    """
    # split categories into seperate categories
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
     # use the first row to extract categories names
    category_colnames = [i[:-2] for i in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #convert categories values to numeric instead of strings
    for column in categories:
        categories[column] = [cat[len(cat)-1:] for cat in categories[column]]
        # convert column from string to numeric
        categories[column] =categories[column].astype(int)
        
        #pd.Series(categories[column], dtype="int64")
    
    # drop categories column in df 
    df.drop(columns = ['categories'], inplace=True)

    # Merge the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #df = df.join(categories)

    #remove duplicates
    df.drop_duplicates(inplace=True)

    print("Duplicate Count=", df.duplicated().sum())
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.

    Args:
        df (pandas.DataFrame): Cleaned dataframe.
        database_filename (str): Filepath for the output SQLite database.
        
    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('categories', engine, if_exists='replace', index=False)


def main():
    """
    Main function to orchestrate the data processing pipeline.

    Reads command line arguments, loads data, cleans it, and saves it to a database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()