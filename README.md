# Disaster Response Web App

## Installation
The code contained in this repository was written in HTML and Python 3, and requires the following Python packages: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle, warnings.

## Project Overview
This repository contains code for a web app which an emergency worker could use during a disaster event (e.g. an earthquake or hurricane), to classify a disaster message into several categories, in order that the message can be directed to the appropriate aid agencies. 

The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new datasets for model training purposes.

## File Descriptions
* **process_data.py**: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
* **train_classifier.py**: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
* **ETL Pipeline Preparation.ipynb**: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.
* **ML Pipeline Preparation.ipynb**: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which algorithm to use. train_classifier.py effectively automates the model fitting process contained in this notebook.
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: This folder contains all of the files necessary to run and render the web app.

## Running Instructions
### ***Run process_data.py***
1. Save the data folder in the current working directory and process_data.py in the data folder.
2. From the current working directory, run the following command:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### ***Run train_classifier.py***
1. In the current working directory, create a folder called 'models' and save train_classifier.py in this.
2. From the current working directory, run the following command:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### ***Run the web app***
1. Save the app folder in the current working directory.
2. Run the following command in the app directory:
    `python run.py`
3. Go to http://0.0.0.0:3001/

## Screenshots

***Screenshot 1: App Front Page***
![Screenshot 1](https://github.com/gkhayes/disaster_response_app/blob/master/Screenshot1.JPG)

***Screenshot 2: App Results Page***
![Screenshot 2](https://github.com/gkhayes/disaster_response_app/blob/master/Screenshot2.JPG)

## Warning
The datasets included in this repository are very unbalanced, with very few positive examples for several message categories. In some cases, the proportion of positive examples is less than 5%, or even less than 1%. In such cases, even though the classifier accuracy is very high (since it tends to predict that the message does not fall into these categories), the classifier recall (i.e. the proportion of positive examples that were correctly labelled) tends to be very low. As a result, care should be taken if relying on the results of this app for decision making purposes.

## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
