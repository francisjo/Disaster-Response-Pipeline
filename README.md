# Disaster Response Pipeline Project
##### Table of Contents 

1. [Installation](#installation)  
2. [Motivation](#motivation)  
3. [Project Descriptions](#ProjectDescriptions)
4. [Files Descriptions](#FilesDescriptions)
5. [Instructions](#instructions)


## Installation <a name="installation"/>
##### The following packages are required:
- pyplot
- pandas
- numpy
- sklearn
- re
- sys
- json
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3


## Motivation: <a name="motivation"/>
The goal of the project is creating a machine learning pipeline of disaster messages to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## Project Descriptions: <a name="ProjectDescriptions"/>
##### The Project is divided in the following Sections:

- Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
- Machine Learning Pipeline to train a model able to classify text message in categories
- Web App to show model results in real time.

## Files Descriptions: <a name="FilesDescriptions"/>
##### The files structure is arranged as below:
    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app

    - data
    |- disaster_categories.csv  # data to process 
    |- disaster_messages.csv  # data to process
    |- process_data.py
    |- InsertDatabaseName.db   # database to save clean data to

    - models
    |- train_classifier.py 

    - README.md

### Instructions:<a name="instructions"/>
 1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

 2. Run the following command in the app's directory to run your web app.
    `python run.py`

 3. Go to http://0.0.0.0:3001/ or http://localhost:3001/


