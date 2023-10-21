# Email Spam Classifier

This GitHub project is dedicated to building an Email Spam Classifier using natural language processing techniques. The project comprises three Jupyter notebooks that cover the entire process, from preprocessing email data to training a spam classifier and testing its performance.

## Project Structure

The project is divided into three main Jupyter notebooks:

### 1. Preprocessing (Preprocessing.ipynb)

In this notebook, the raw email corpus is processed and transformed into a test and training dataset. The preprocessing phase is crucial for extracting meaningful features from the email content and preparing the data for machine learning. Key libraries used in this notebook include:

- **NLTK (Natural Language Toolkit)**: Used for text tokenization, stopword removal, and other text processing tasks.
- **Beautiful Soup 4 (bs4)**: Employed for HTML parsing to clean up HTML content within emails.
- **Numpy**: Used for various array and matrix operations.
- **Pandas**: Utilized for data manipulation and organization.

The primary tasks in this notebook involve text cleaning, tokenization, and feature extraction from the email content, which are essential for training a spam classifier.

### 2. Training (Training.ipynb)

The training notebook focuses on building a spam classifier model. Notably, this model is created without relying on machine learning libraries like scikit-learn. Instead, it leverages fundamental data manipulation and linear algebra techniques. Key aspects of this notebook include:

- **Numpy**: Utilized for matrix operations, especially in implementing mathematical aspects of the classifier.
- **Pandas**: Used for data organization, exploration, and preprocessing.
- Custom-coded algorithms for training the spam classifier.
- Model evaluation metrics for assessing classifier performance.

The primary goal of this notebook is to create an effective spam classifier based on the features extracted during the preprocessing phase.

### 3. Testing (Testing.ipynb)

In the testing notebook, the performance of the spam classifier developed in the training notebook is evaluated. This evaluation is done on a separate dataset not seen during training to ensure the model's generalizability. Key aspects of this notebook include:

- **Numpy**: Used for matrix operations related to the testing phase.
- **Pandas**: Utilized for data organization and the preparation of test data.
- Model evaluation metrics to assess how well the classifier performs on unseen data.

## How to Use This Repository

1. Clone this repository to your local machine using Git.

   ```bash
   git clone https://github.com/APTUUU/Email-Spam-Detector.git
   ```

2. Open each of the Jupyter notebooks (Preprocessing.ipynb, Training.ipynb, Testing.ipynb) using a Jupyter notebook environment or JupyterLab.

3. Follow the step-by-step instructions in the notebooks to understand the project, preprocess email data, train a spam classifier, and test its performance.

4. Feel free to modify and experiment with the code to adapt it to your specific use case or improve the classifier's performance.

## Requirements

To run the Jupyter notebooks in this project, you will need to have the following libraries and dependencies installed:

- Python (3.6 or higher)
- Jupyter Notebook
- NumPy
- Pandas
- NLTK
- Beautiful Soup 4 (bs4)

You can install these dependencies using `pip` or your preferred package manager.

```bash
pip install numpy pandas nltk beautifulsoup4
```

## Contribute

Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to create issues or pull requests.

