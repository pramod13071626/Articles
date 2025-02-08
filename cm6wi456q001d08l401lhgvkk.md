---
title: "How to Build an AI Resume Screener Using Python and Machine Learning"
seoTitle: "Creating an AI Resume Screener with Python"
seoDescription: "Learn to build an AI-powered resume screener using Python and machine learning for efficient recruitment automation"
datePublished: Sat Feb 08 2025 18:00:48 GMT+0000 (Coordinated Universal Time)
cuid: cm6wi456q001d08l401lhgvkk
slug: how-to-build-an-ai-resume-screener-using-python-and-machine-learning
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1739035095305/98ad271f-5439-4754-949a-bf217c9de544.jpeg
tags: ai, python, machine-learning, resume-screening

---

In today's competitive job market, recruiters receive hundreds of resumes for each open position, making **manual screening time-consuming** and inefficient. ***AI-powered resume*** screeners can help **automate** this process by evaluating resumes, extracting relevant information, and ranking candidates based on job parameters.

### ***Why is resume screening done?***

The precision of job-matching models is significantly affected by data quality. Inadequately formatted resumes can lead to ***misclassifications***, which may affect hiring choices.

Since resumes include ***unstructured textual information***, preprocessing is crucial to transform disorganized data into a format appropriate for machine learning algorithms.

This blog will guide you through building an ***AI resume screener*** using Python and machine learning.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdksh9xRXXS9lwW1vmtAO7J0KBICj1yvY8Uxil1P67VTnYvTjRXCIbmAqVJpo7KMjZYi2LXHsLEaroeh9G2nYOEZjiHdJASI6z2QtmkhuTyQkPtyYJ-QRj8DsWXZ6_irZ4qNo6TGQ?key=rUAyOYnCo0CtX7x8Vptk78op align="left")

* ## ***Prerequisites:***
    

Before we begin, ensure you have the following:

* ***Python*** installed (version 3.x)
    
* ***Basic understanding*** of machine learning and Natural Language Processing (NLP).
    
* ***Libraries:*** pandas, numpy, sci-kit-learn, ***spaCy***, nltk, and ***Flask*** (for deployment).
    
* ## ***Environment Setup:***
    

Install the required Python libraries using pip:

**!pip install pandas numpy scikit-learn spacy nltk flask**

Download the English language model for spaCy:

**!python -m spacy download en\_core\_web\_sm**

This sets up the ***pre-trained spaCy*** model for text processing, which includes ***tokenization, lemmatization,*** *and* ***named entity recognition (NER)***.

* ## ***Collecting and Preprocessing Data:***
    

We need a dataset containing resumes and job descriptions to build an effective ***AI resume screener***. You can use ***public datasets*** or collect resumes in a structured format ***(e.g., CSV, JSON)****.*

**Kaggle Dataset:** [***<mark>https://www.kaggle.com/code/gauravduttakiit/resume-screening-using-machine-learning</mark>***](https://www.kaggle.com/code/gauravduttakiit/resume-screening-using-machine-learning)

* **Loading a Sample Dataset:**
    

> ```python
> import pandas as pd
> 
> # Loading the Data Set
> 
> df = pd.read_csv(r"/content/UpdatedResumeDataSet.csv")
> 
> df.head()
> ```

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdtl8dEn9fploXRp-Ocnf23UoP79AZAhBALGYJzofpZhkr0PJThNN7jkOWX0lByEFyu-jfNkCabfpD4UHL3FJwT8YzTh9BSGK8o2_whWCUdG3a6q2isR9biQmaU9-GDzTncHqkyWw?key=rUAyOYnCo0CtX7x8Vptk78op align="left")

* ### ***Text Preprocessing***
    

Text preprocessing refers to a series of techniques used to ***clean, transform***, and ***prepare raw textual data*** into a format that is suitable for NLP or ML tasks.

Text preprocessing plays an essential role in developing an AI-based resume screening tool. Resumes consist of unstructured text filled with noise (like ***special characters, irrelevant words, and inconsistent formatting***), so it's vital to ***clean and preprocess*** the data before inputting it into a machine-learning model.

```python
import spacy

import re

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):

text = re.sub(r'[^a-zA-Z ]', '', text)

doc = nlp(text.lower())

return " ".join([token.lemma_ for token in doc if not token.is_stop])

df['Cleaned_Resume'] = df['Resume'].apply(preprocess_text)

df.head()
```

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcim0oxnfaEUqYME-35TkNQHeot2YLrvCzqR1wyMKPmjUV6oiyL0ggLRTImSeAWhGwcjrsTqy2J9zR09CXZUPW_Ib1LGbmmhcsrDnB7768epuimrixsf06t7zLodI0mxOvBc2GZYg?key=rUAyOYnCo0CtX7x8Vptk78op align="left")

* ## ***Feature Extraction Using TF-IDF (Term Frequency-Inverse Document Frequency)***
    

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical method utilized in Natural Language Processing (NLP) and ***Information Retrieval*** to assess the significance of a word within a document in relation to a ***set of documents*** ***(corpus)***. It helps in pinpointing important keywords while

***Term Frequency (TF) measures*** how often a word appears in a document.

***Inverse Document Frequency (IDF):*** Reduces the weight of commonly used words by measuring how rare a word is across all documents.

***TF-IDF Score:*** The final weight of a word in a document is computed as:

### ***<mark>TF−IDF=TF×IDF</mark>***

We convert text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).

* ```python
      from sklearn.feature_extraction.text import TfidfVectorizer
      
      vectorizer = TfidfVectorizer(max_features=3000)
      
      X = vectorizer.fit_transform(df['Cleaned_Resume'])
      
      y = df['Category']  # Target labels
    ```
    
* ## ***Building a Machine Learning Model***
    

A machine learning (ML) model represents a mathematical formulation that learns from ***data patterns*** to make ***predictions*** or decisions without explicit programming. It comprises:

***Input Data:*** The information utilized for training/testing.

***Training Algorithm:*** The approach that identifies patterns within the data.

***Output Predictions:*** The decisions made by the model are based on learned patterns.

These are three categories of machine learning models, as follows:

1. **Supervised Learning (e.g., Classification, Regression)**
    
2. **Unsupervised Learning (e.g., Clustering, Anomaly Detection)**
    
3. **Reinforcement Learning (e.g., Decision-Making in AI)**
    

We train a classification model to categorize resumes based on their content.

### ***Why Random Forest Classifier?***

A Random Forest Classifier is an ensemble learning method that ***constructs multiple decision trees*** and integrates their outcomes for more precise and stable predictions.

### Functional ***Process:***

Generates several decision trees using various subsets of the data.

Each tree provides a ***prediction*** (classification).

The final ***result*** is established through a ***majority vote*** (the most frequent prediction).

```python
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Evaluate the model

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

### ***<mark>Accuracy achieved: 98.45%</mark>***

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXf5eWsgVQJz2eCk9MN7yxLD0mKfx8Jeg2IEFPMpWmDt5I-WI1a1jZrj6ZJzkvd0fkfvc_zZik6eLTlfyWYfWHGWFFoGYfo1qyO1jYZa1yui4jQYbGNtc4uDslRC3ubyJ5uQ4LKv?key=rUAyOYnCo0CtX7x8Vptk78op align="center")

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdKP0gTg3JVZ99Jhimi2odGsnzJtb1oApN1Vi-XJPwfkm5jybN-pyeXfKv0q6FPEpmTAfU18_L8cfKfloaxGl0Huud4aCcYmaaVOvncDy1RSue2xUOgaNj-Uy58LE2_fXe0yjD-5w?key=rUAyOYnCo0CtX7x8Vptk78op align="center")

* ![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXehOlVwr759l3U1DnS0Lr2w5ngbtsi1zulqenRc1-fiQMg-8g-5O0wFmvhJfOWl4OY4TwLr0Ni2TLclkrtXSPlGjBedopgp_kk98q0hYTOxRGMl-UWd6FQ0xyPHkU4DwcxwZUKIYQ?key=rUAyOYnCo0CtX7x8Vptk78op align="left")
    
    ![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcLdvU7vh8-3iHCLuICA5Wstv_1t88SnGFSu_9q_JsnR7MJQ3r8KMNO7CD1RtXYLtzRwY8jJvT_SUz2hv72RgbbgAQkdcBN1xvCrYCUV1CuchfsSU9guliu6tJLz7-zFMZ6HYJkgA?key=rUAyOYnCo0CtX7x8Vptk78op align="center")
    
* ## ***Deploying the AI Resume Screener:***
    

Flask is a minimalistic ***web framework*** for Python that facilitates the development of ***web applications***.

### ***Creating a Flask API:***

The Flask API acts as the ***backend*** for our ***AI Resume Screener***, enabling users to upload resumes and receive predictions regarding job categories.

***Storage of Model and Vectorizer:*** The Random Forest model and the ***TF-IDF vectorizer*** that have been trained are saved using pickle to prevent the need for retraining.

***API Endpoint (/predict):*** It accepts a ***POST request*** containing a resume in JSON format. The resume undergoes preprocessing and vectorization prior to being input into the model for its prediction.

***Response to Prediction:*** The model determines the job category and returns the result as a ***JSON response.***

**Starting the API:** The Flask application operates on ***localhost:5000***, allowing for integration with ***web or job porta****l* applications.

```python
from flask import Flask, request, jsonify

import pickle

app = Flask(__name__)

with open("model.pkl", "wb") as f:

pickle.dump((model, vectorizer), f)

@app.route('/predict', methods=['POST'])

def predict():

data = request.json['resume']

processed_resume = preprocess_text(data)

vectorized_resume = vectorizer.transform([processed_resume])

prediction = model.predict(vectorized_resume)[0]

return jsonify({'category': prediction})

if name == '__main__':

app.run(debug=True)
```

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcAUR2WZM-uvVVnKijduYPVtziUolutskkkZMFu8_YyjreqcqAR2nmUtVmtsgKg8yvCOc4AUH3aYgC0aTmQIi6Ogm79X66XdQwFuPhD4ExdhhLCtBVjSXMfASJ3uIcL5LRG1vxEnw?key=rUAyOYnCo0CtX7x8Vptk78op align="left")

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcjdnILxs7PyXX5u3mrrbI3zmEoxUpyv3fnZcllD4GDn22pupohud7yBTN2MOOo5OpEMYYsWO-o1KjD2I-hfWOoklSFS1K_bmUsZ8BUVCFKqB0ekqa_0jXfXrHH7PUIhN1J1FRu9w?key=rUAyOYnCo0CtX7x8Vptk78op align="left")

Building an AI-powered resume screener can significantly streamline the hiring process by ***automating resume classification***. This project covers essential steps, including data preprocessing, feature extraction, ***model training***, and ***deployment***. You can enhance the screener by integrating ***Named Entity Recognition (NER)*** for extracting candidate details and refining the model with ***deep learning techniques.***

I hope it will be a useful article for you. Happy coding 🤞

Contact Accounts: [Linkedin](http://www.linkedin.com/in/pramoddeore162633).