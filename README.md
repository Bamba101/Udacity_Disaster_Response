# Disaster Response Pipeline Project

Project build a model for an API that classifies disaster messages.
This can be used by organisations to easly identify the nature of the disaster so as to reach out for a relevant relief agency (Fire , Medical , Security etc)

Project has 3 main parts:
1. creating an ETL pipeline to clean the data.
2. Machine Learning pipeline to train a model.
3. Web app to predict new messages.

## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Instructions](#Instructions)

## Description
Project Three Python scripts:

- **`data/process_data.py`**: Creates ETL Pipeline that Loads data from an SQLite database Clean it and save.
- **`models/train_classifier.py`**: Builds Machine Learning Model to Classify disasters messages
- **`app/run.py`**: Runs a web app then Classify disasters messages using the trained ML model.

## Installation
To run this project, you need Python 3.x and below libraries:

- pandas
- numpy
- scikit-learn
- sqlalchemy
- nltk
- flask

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn sqlalchemy nltk flask
```

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Then, open a web browser and go to http://localhost:3000/ to access the web app.


