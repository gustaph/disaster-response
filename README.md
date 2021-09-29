### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Acknowledgment](#acknowledgment)

## Project Overview <a name="overview"></a>

This project consists of preparing text messages about real-life disasters, pre-labeled by [Figure Eight](https://appen.com), to build a supervised learning model. In short, you type a message and the program sorts it according to the 36 available labels, specifying the problem by different types of organization.

For the supervised model building, I chose [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) over [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) because of the higher speed in making the prediction (having almost the same accuracy as RF). LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages [[1]](https://www.kaggle.com/prashant111/lightgbm-classifier-in-python).

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel and GPU learning.
- Capable of handling large-scale data.


![overview](https://user-images.githubusercontent.com/54601061/135074628-31b69cc6-0c81-4f8f-9715-df94d79165ae.gif)

---

## Installation <a name="installation"></a>

For development, the following dependencies were used:

- [Pandas](https://pandas.pydata.org)
- [Numpy](https://numpy.org)
- [Sklearn](https://scikit-learn.org/stable/)
- [NLTK](https://www.nltk.org/data.html)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- [SQLAlchemy](https://www.sqlalchemy.org)
- [Flask](https://flask.palletsprojects.com/en/2.0.x/)
- [Plotly](https://plotly.com)

## File Descriptions <a name="files"></a>

The following are the files available in this repository:

- `notebooks/ETL Pipeline Preparation.ipynb` - data cleaning processing steps.
- `notebooks/ML Pipeline Preparation.ipynb` - steps in creating the supervised model and its evaluation.
- `data/process_data.py` - data cleaning pipeline (load, merge, clean and, store the data).
- `data/disaster_categories.csv` - data related to disaster categories.
- `data/disaster_messages.csv` - data related to disaster messages.
- `data/DisasterResponse.db` - database containing the result of the ETL pipeline.
- `models/train_classifier.py` - machine learning pipeline (load, split, train and tune a model, export the final model).
- `models/classifier.pkl` - the supervised model.
- `app/run.py` - launch the Flask app.
- `app/templates/*` - html templates for web application.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (*or to the link that will appear in the console*)

The files in the `notebooks` folder do not have any practical connection with the progress of the application, the files were used to test and study the structure of the data sets.

## Acknowledgment <a name="acknowledgment"></a>

Must give credit to [**Udacity**](https://www.udacity.com) for providing the project templates and [**Figure Eight Inc.**](https://appen.com) for providing the data.
