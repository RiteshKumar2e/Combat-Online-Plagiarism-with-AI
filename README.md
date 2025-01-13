# Plagiarism Detection and Paraphrase Check

## Overview

This project implements a **Plagiarism Detection System** using machine learning, NLP, and a **Paraphrase Detection** mechanism via an external API. The primary goal of the system is to determine whether a given text is plagiarized by analyzing it in relation to a dataset of original and plagiarized content. Additionally, it integrates a paraphrasing API to check if the input text has been reworded or paraphrased. The system leverages a machine learning model to predict whether the text is plagiarized and visualizes the results through a cosine similarity measure.

## Project Description

Plagiarism detection is essential in academic, legal, and content creation fields to ensure originality and avoid intellectual property infringement. This project focuses on building a **Text Classification Model** that detects plagiarism in a text by comparing it to a dataset of original and plagiarized content. The dataset consists of two columns: the original source text and the potentially plagiarized text.

The project involves the following key steps:

1. **Data Preprocessing**: Clean and preprocess the text by removing punctuation, converting to lowercase, and removing stopwords.
2. **Feature Extraction**: Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform the text into numerical features.
3. **Model Training**: Train a **Logistic Regression** model to classify text as plagiarized or not.
4. **Paraphrase Detection**: Use an external paraphrasing API to check if the text has been reworded.
5. **Prediction and Similarity Measurement**: Predict whether a given text is plagiarized using the trained model and calculate the **cosine similarity** between the input text and the training data to assess the level of plagiarism.
6. **Visualization**: Display the results in a bar chart to visualize the plagiarism level based on the cosine similarity score.

## Prerequisites

Before running the project, ensure that you have the following Python packages installed:

- `requests`: For making HTTP requests to the Paraphrasing API.
- `nltk`: Natural Language Toolkit, used for text preprocessing (e.g., removing stop words).
- `sklearn`: Used for machine learning, including model training and evaluation.
- `pandas`: Data manipulation and analysis, used for loading and processing the dataset.
- `joblib`: For saving and loading the trained model.
- `matplotlib`: For visualizing the plagiarism detection results.
- `seaborn`: For enhanced data visualization.

### Installing the Dependencies

You can install the necessary dependencies using the following pip command:

```bash
pip install requests nltk scikit-learn pandas joblib matplotlib seaborn
