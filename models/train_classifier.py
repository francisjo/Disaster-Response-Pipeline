# import libraries
import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('disaster_messages_table',engine)
    X = df['message']  # Message Column
    y = df.iloc[:, 4:] # Classification label
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    Function: split text into words and return the root form of the words
    Input:
        text(str): the message
    Output:
        lemm(list of str): a list of the root form of the message words
    '''
    # Normalize Text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    words = word_tokenize(text)
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w, pos='v').strip() for w in lemmed]
    return lemmed

def build_model():
    moc = MultiOutputClassifier(RandomForestClassifier())
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
        y_pred = model.predict(X_test)
        results = pd.DataFrame(columns=['Category', 'f1_score', 'precision', 'recall'])
        num = 0
        for cat in category_names:
            dictreport = classification_report(Y_test[cat], y_pred[:,num], output_dict=True)
            #print(type(dictreport))
            results.set_value(num+1, 'Category', cat)
            results.set_value(num+1, 'f1_score', dictreport['weighted avg']['f1-score'])
            results.set_value(num+1, 'precision', dictreport['weighted avg']['precision'])
            results.set_value(num+1, 'recall', dictreport['weighted avg']['recall'])
            num += 1
        print(results)
        print('Aggregated f_score:', results['f1_score'].mean())
        print('Aggregated precision:', results['precision'].mean())
        print('Aggregated recall:', results['recall'].mean())
        
        


def save_model(model, model_filepath):
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        save_model(model, model_filepath)
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
