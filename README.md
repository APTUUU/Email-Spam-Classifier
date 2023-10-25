# Email Spam Filter

This repository contains code for an email spam filter. It includes Jupyter notebooks for preprocessing, training, and testing the spam filter. Below is a brief overview of each notebook's contents.

- [Prerequisites](#prerequisites)
- [Pre-Processing](#pre-processing)
  - [Imports](#imports)
  - [Constants](#constants)
  - [Reading and Loading Email Data](#reading-and-loading-email-data)
  - [Vocabulary Generation](#vocabulary-generation)
  - [Feature Extraction](#feature-extraction)
- [Training](#training)
  - [Imports](#imports-1)
  - [Constants](#constants-1)
  - [Read and Load Features](#read-and-load-features)
  - [Training Naive Bayes Model](#training-naive-bayes-model)
- [Testing, Inference & Evaluation](#testing-inference-evaluation)
  - [Imports](#imports-2)
  - [Constants](#constants-2)
  - [Load Data](#load-data)
  - [Set Prior](#set-prior)
  - [Making Predictions](#making-predictions)
  - [Visualizing Results](#visualizing-results)
- [Usage](#usage)

## Prerequisites

- Python (>= 3.6)
- NumPy
- Pandas
- NLTK
- Matplotlib
- Seaborn
- BeautifulSoup
- WordCloud

Ensure that you have the above prerequisites installed before using the Email Spam Filter code. You can install them using the following commands:

```bash
pip install numpy pandas nltk matplotlib seaborn beautifulsoup4 wordcloud
```

## Pre-Processing

The pre-processing notebook focuses on preparing the email data for training and testing the spam filter. Here are the key steps:

### Imports
- Import necessary Python libraries, including NumPy, pandas, and NLTK (Natural Language Toolkit).

### Constants
- Define file paths and constants used throughout the code, including the paths to data files and vocabulary size.

### Reading and Loading Email Data
- Load the email data from various directories, including spam and ham (non-spam) emails.
- Clean and preprocess the data, removing HTML tags, tokenizing, and stemming.

### Vocabulary Generation
- Create a vocabulary based on the most frequent words in the dataset.
- Save the vocabulary as a CSV file.

### Feature Extraction
- Convert the email data into a sparse matrix representation.
- Save the training data as a text file.

## Training

The training notebook covers the process of training a Naive Bayes model for spam classification. Here are the main sections:

### Imports
- Import required libraries, including pandas and NumPy.

### Constants
- Define constants, such as file paths and the vocabulary size.

### Read and Load Features
- Load the training data and store it in a pandas DataFrame.

### Training Naive Bayes Model
- Calculate the probability of spam and total word count in each category (spam and ham).
- Calculate the probability of each word occurring in spam and ham emails.
- Save the trained data to a CSV file.

## Testing, Inference & Evaluation

The testing notebook focuses on evaluating the spam filter's performance and making predictions on new emails. Here are the key sections:

### Imports
- Import necessary libraries, including pandas and NumPy.
- Use seaborn and matplotlib for visualizations.

### Constants
- Define constants, such as file paths and the vocabulary size.

### Load Data
- Load the test data and probability data from CSV files.

### Set Prior
- Calculate the joint probabilities of emails being spam or ham based on the training data.

### Making Predictions
- Make predictions for spam or ham based on the joint probabilities.
- Calculate accuracy, false positives, false negatives, recall, precision, and F1-score.

### Visualizing Results
- Visualize the results using plots, decision boundaries, and word clouds.

## Usage

You can use these notebooks to preprocess, train, and test a spam filter for email classification. The code allows you to experiment with different email datasets and parameters to improve classification accuracy. Please refer to the specific notebooks for detailed code and instructions.

For more information and to access the complete code, please visit the [Email Spam Detector Repository](https://github.com/APTUUU/Email-Spam-Detector).
