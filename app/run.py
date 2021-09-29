import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # 1st plot -----------
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 2nd plot -----------
    total_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    total_labels = list(total_counts.index)

    # 3rd plot -----------
    df_melt = pd.melt(df.drop(['id', 'message', 'original'], axis=1), id_vars=['genre'])
    valid_counts = df_melt[df_melt['value'] == 1].groupby(['genre', 'variable'])['value'].value_counts()

    top_news = valid_counts['news'].sort_values(ascending=False).head(10)
    labels_news = list(top_news.index.get_level_values(0))
    values_news = list(top_news.values)

    top_direct = valid_counts['direct'].sort_values(ascending=False).head(10)
    labels_direct = list(top_direct.index.get_level_values(0))
    values_direct = list(top_direct.values)

    top_social = valid_counts['social'].sort_values(ascending=False).head(10)
    labels_social = list(top_social.index.get_level_values(0))
    values_social = list(top_social.values)

    standard = {
        "hoverinfo": "label+percent+value",
        "hole": .4,
        "type": "pie"
    }

    data_social = {
        "values": values_social,
        "labels": labels_social,
        "domain": {"column": 0},
        "title": "Social"
    }
    data_social.update(standard)

    data_direct = {
        "values": values_direct,
        "labels": labels_direct,
        "domain": {"column": 1},
        "title": "Direct"
    }
    data_direct.update(standard)

    data_news = {
        "values": values_news,
        "labels": labels_news,
        "domain": {"column": 2},
        "title": "News"
    }
    data_news.update(standard)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, {
            'data': [
                Pie(data_social),
                Pie(data_direct),
                Pie(data_news)
            ],

            # (REFERENCE) Plotly - Bar Chart and Pie Chart [https://www.tutorialspoint.com/plotly/plotly_bar_and_pie_chart.htm]
            'layout': {
                "title": "Most common categories by genre",
                "grid": {"rows": 1, "columns": 3}
            }
            
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()