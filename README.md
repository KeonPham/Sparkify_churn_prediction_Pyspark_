# Sparkify User Churn Prediction

This Jupyter notebook contains a comprehensive analysis and machine learning pipeline to predict user churn for the fictional music streaming service, Sparkify. The goal is to build a predictive model that identifies users likely to churn based on their behavior and usage patterns.

Here is my blog post on medium: https://medium.com/@khoapha/predicting-user-churn-with-pyspark-a-comprehensive-guide-5210b401daa2

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)

## Introduction

In this project, we aim to predict which users are likely to churn using their activity logs from the Sparkify app. Churn prediction is crucial for understanding user behavior and taking necessary actions to retain users.

## Dataset

The dataset consists of user activity logs, including actions such as playing songs, liking songs, downgrading accounts, and more. The data is provided in a JSON format, and it includes the following fields:

- `userId`: Unique identifier for each user
- `sessionId`: Identifier for each session
- `page`: Action performed by the user
- `song`: Song played by the user
- `artist`: Artist of the song
- `length`: Duration of the song played
- `ts`: Timestamp of the action

## Data Preprocessing

In this section, we load the dataset and perform necessary preprocessing steps, including:

- Remove null & empty data of userId & sessionId
- Removing duplicate entries


## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the distribution of the data and identify key trends and patterns. Visualizations and summary statistics are used to explore user behavior of Active and Inactive groups.

## Feature Engineering

Feature engineering involves creating new features from the existing data that can improve the predictive power of the model. Features include:

- Average Song Length (avg_song_length): The average length of all songs played by the user. This feature helps understand how long users typically engage with the content.

- Total Unique Artists (total_artist): The total number of unique artists the user has listened to. This indicates the diversity of the user's listening habits.

- Total Unique Songs (total_song): The total number of unique songs the user has played. This provides a measure of the user's engagement with different tracks.

- Total Sessions (total_session): The total number of actions performed by the user (indicated by the page column). This feature captures the user's overall activity on the platform.

- Total Thumbs Up (total_thumb_up): The total number of times the user has given a thumbs up. This reflects the user's positive feedback and satisfaction.

- Total Thumbs Down (total_thumb_down): The total number of times the user has given a thumbs down. This indicates the user's negative feedback and potential dissatisfaction.

- Total Add Friend (total_add_friend): The total number of times the user has added a friend. This feature can provide insight into the user's social behavior on the platform.

- Total Add to Playlist (total_add_playlist): The total number of times the user has added a song to a playlist. This reflects the user's engagement in curating and managing their personal playlists.

- Gender (gender): The gender of the user, mapped to numerical values (0 for Male, 1 for Female). This demographic feature can help understand the influence of gender on user behavior and churn.

## Model Building

We use the following machine learning algorithms to build the churn prediction model:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosted Trees (GBT) Classifier
- Hyper Tunned Random Forst Classifier

## Model Evaluation

Models are evaluated using metrics such as accuracy and F1 score and Accuracy. Cross-validation is used to ensure the robustness of the models.

## Hyperparameter Tuning

Hyperparameter tuning is performed using grid search and cross-validation to find the optimal parameters for the best-performing model of Random Forest. The primary focus is on optimizing the F1 score to balance precision and recall.


# Define a parameter grid for hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth,[5, 10]) \
    .addGrid(rf.numTrees, [20, 50]) \
    .addGrid(rf.minInstancesPerNode, [1, 10]) \
    .addGrid(rf.subsamplingRate, [0.7, 1.0]) \
    .build()

## Conclusion
Here is a final result:

| Model                 | F1 Score | Accuracy |
|-----------------------|----------|----------|
| Logistic Regression   | 58.54%   | 67.74%   |
| Random Forest         | 62.69%   | 70.97%   |
| Gradient Boosting     | 67.05%   | 69.35%   |
| Random Forest Tuned   | 61.64%   | 69.35%   |

We should go with Gradient Boosting Model where it achive the highest F1-Score and a roughly balance Accuracy compared to other models. It could be a based model to further improvement with full dataset. 

## Feature Importance from Gradient Boosting Model

| Feature            | Importance (%) |
|--------------------|----------------|
| total_thumb_down   | 17.58          |
| total_thumb_up     | 14.98          |
| gender             | 12.12          |
| avg_song_length    | 11.67          |
| total_add_friend   | 10.66          |
| total_artist       | 10.22          |
| total_add_playlist | 9.87           |
| total_session      | 9.46           |
| total_song         | 3.44           |


## Dependencies
Ensure you have the following dependencies installed before running the notebook:

Python 3.x
PySpark
pandas
numpy
matplotlib
seaborn

