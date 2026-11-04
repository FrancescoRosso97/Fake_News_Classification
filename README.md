# Fake News Classification - Machine Learning Project (Rosso Francesco - 70/83/65482 - Academic Year 25/26)

## Project Overview
This project aims to classify news articles as Fake or Real using Machine Learning techniques. It's based on the "Fake and real news dataset" available on Kaggle. The classification system uses a Linear Support Vector Machine (Linear SVM) and TF-IDF (Term Frequency–Inverse Document Frequency) to transform textual information into numerical features. The objective is to train, optimize, and evaluate a single classification model capable of distinguishing between Fake and Real news articles based on their textual content.

## Dataset
The dataset contains two CSV files:
- `Fake.csv` – articles labeled as fake;
- `True.csv` – articles labeled as real.

Due to file size constraints, these CSV files are not included in this repository. They can be downloaded directly from Kaggle at the following link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.

Each article contains the following fields:
- `title` – article headline;
- `text` – article content;
- `subject` – news category;
- `date` – publication date.

The `title` and `text` columns were combined to create a single textual input feature.

A binary target variable was defined:
- 0 = Fake;
- 1 = Real.

## Dataset Balancing
A balanced subset of 16000 articles was selected:
- 8000 Fake articles;
- 8000 Real articles.

This balanced distribution ensures that both classes are equally represented in the selected dataset.

The dataset was divided into:
- Training set: 11200 articles (70%);
- Test set: 4800 articles (30%).

## Project Workflow
The project follows these main steps:

### Dataset Loading and Preparation
The two CSV files were loaded and combined into a single dataset. 

Text preprocessing was performed to prepare the articles for classification.

The main operations included:
- Conversion of text to lowercase;
- Removal of punctuation and numbers;
- Concatenation of `title` and `text`;
- Creation of binary class labels.

### Feature Extraction
Textual data was transformed into numerical features using TF-IDF vectorization.

TF-IDF and the classifier were combined into a Scikit-learn Pipeline.

### Classification Model
The selected classification algorithm was Linear Support Vector Machine (Linear SVM), implemented using LinearSVC from Scikit-learn.

Only one classifier was used in this project.

The model was trained on the training dataset and subsequently evaluated on the independent test set.

### Hyperparameter Optimization
Hyperparameter optimization was performed using GridSearchCV with 3-fold cross-validation.

The following values of the regularization parameter C were tested:
- C = 0.1;
- C = 1;
- C = 10.

The best parameter was: C = 1. The best mean cross-validation accuracy was 98.60%.

### Model Evaluation
The trained model was evaluated on the test set using:
- Accuracy;
- Precision;
- Recall;
- F1-score;
- Confusion Matrix.

Both macro and weighted averages were calculated for the classification metrics.

### Error Analysis
The incorrectly classified articles were identified and analyzed by comparing the predicted labels with the actual labels.

The model produced:
- 28 Fake articles incorrectly classified as Real;
- 22 Real articles incorrectly classified as Fake.

A total of 50 articles were incorrectly classified.

### Data visualization
Data visualization techniques were used to explore the distribution of Fake and Real news articles.

The visualizations include:
- Class distribution bar chart;
- Hyperparameter optimization plot;
- Confusion Matrix;
- Two-dimensional t-SNE visualization.

For the t-SNE visualization, a balanced sample of 2000 articles was selected, containing 1000 Fake and 1000 Real articles.

TF-IDF features were first reduced using TruncatedSVD, followed by t-SNE to obtain a two-dimensional representation.

The visualization allows the distribution of the two classes to be explored without introducing additional classification models.
