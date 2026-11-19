# Fake News Classification - Machine Learning Project (Rosso Francesco - 70/83/65482 - Academic Year 25/26)

## Project Overview
This project focuses on the binary classification of news articles as Fake News or Real News based on their textual content. The classification system uses a Support Vector Machine (SVM) with a linear kernel, combined with text preprocessing, TF-IDF features extraction, features scaling, and Principal Component Analysis (PCA).

## Project Workflow
The project follows these main steps.
### Dataset Loading
The dataset used is the Fake and Real News Dataset, available on Kaggle. It consists of two CSV files:
- `Fake.csv` – articles labeled as fake;
- `True.csv` – articles labeled as real.

Due to file size constraints, these CSV files are not included in this repository. They can be downloaded directly from Kaggle at the following link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset.

Each article contains the following fields:
- `title` – article headline;
- `text` – article content;
- `subject` – news category;
- `date` – publication date.

The two classes are encoded as follows:
- `0` = Fake;
- `1` = Real.

### Dataset Preparation
A balanced subset of `16.000` articles is selected, containing `8.000` fake news articles and `8.000` real news articles. This balanced distribution ensures that both classes are equally represented in the selected dataset. 

The `title` and `text` columns of articles are combined to create a single textual input feature.

### Text Preprocessing
Text preprocessing is performed to prepare the articles for classification. It includes:
- Conversion to lowercase;
- Removal of text enclosed in square brackets;
- Removal of punctuation;
- Removal of words containing numerical digits.

### TF-IDF Features Extraction

Text documents are transformed into numerical representations using TF-IDF. The vectorizer uses:
- English stop-word removal;
- max_df = `0.7`;
- min_df = `5`;
- Unigrams and bigrams;
- Maximum of `2.000` features.

### Features Scaling

MinMaxScaler is applied to scale the extracted features.

### Dimensionality Reduction

Principal Component Analysis (PCA) reduces the features space to `50` principal components.

### Classification

A Support Vector Machine with a linear kernel is used for binary classification. The regularization parameter `C` is optimized using GridSearchCV with the following candidate values:

`[0.01, 0.1, 1, 10 100]`

### Model Validation

Five external ShuffleSplit partitions are generated, using:

- `70%` of the dataset for training (`11200` articles);

- `30%` of the dataset for testing (`4800` articles).

An internal three-fold cross-validation procedure selects the regularization parameter `C`.

### Performance Evaluation

The model is evaluated using:

- Accuracy for each external split;
- Mean accuracy and standard deviation;
- Precision for each class;
- Recall for each class;
- F1-score for each class;
- Support for each class;
- Confusion matrix.

Class-specific metrics and the confusion matrix aggregate predictions from the five external test splits. Since ShuffleSplit can select the same article in multiple test sets, these predictions do not necessarily correspond to unique articles.

### Data Visualization

A confusion matrix is used to visualize correct and incorrect predictions.

Additionally, a pairplot displays the relationships among the first three principal components, using the training samples from the first external split.

The pairplot is intended for exploratory visualization of the features distribution and does not directly measure SVM classification performance.
