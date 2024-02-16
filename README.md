# Sucide Intent Analysis - Natural Language Process Project
This project develops **machine learning model** to analyze online textual data for suicide risk **assessment**. Utilizing datasets derived from Reddit forums, the study evaluates the efficacy of various algorithms, including K-Neighbors Classifier, Multinomial Naïve Bayes, and Logistic Regression, to discern suicidal ideation. The optimal model demonstrated a test accuracy of 69.18% with an AUC score of 0.77, highlighting the potential of natural language processing in mental health diagnostics and indicating avenues for future refinement through advanced learning methodologies.  

## Dataset 
The dataset is from the Reddit website "[Peer support for anyone struggling with suicidal thoughts](https://www.reddit.com/r/SuicideWatch/ "悬停显示")" community and "[depression, because nobody should be alone in a dark place](https://www.reddit.com/r/depression/ "悬停显示")" community. There are 1957 items in the dataset that have been manually classified with 6 variables. The training dataset has 7613 observations, whereas the test dataset contains the remaining observations.
<img src="/pic/Screen Shot 2024-02-14 at 5.53.51 PM.png" width = "700" height = "400" alt="cmt" />
## Methodology(Natural Language Processing)
* *Pre-processing*  
Data preprocessing is a critical step, ensuring text data's consistent and digestibility for NLP model construction. Techniques include converting text to **lowercase** for consistency, **tokenizing** characters into discrete components, **verb lemmatization** for meaningful analysis, **removing stopwords and punctuation** to focus on relevant information. This phase transforms raw text into a structured format, enabling precise subsequent analysis​.

* *Text Analysis*  
Text analysis employs vectorization methods such as CountVectorize, TF-IDF transfer, and HashingVectorize to translate filtered text into numerical vector data. **CountVectorize** quantifies word occurrences, **TF-IDF** adjusts word counts based on their document frequency to highlight important terms, and **HashingVectorize** maps words to numerical amount, optimizing memory usage. These methodologies convert text into fixed-length feature vectors suitable for machine learning algorithms​.  

* *Models and Algorithms*  
This study explores Multinomial Naive Bayes, K-Neighbors Classifier, and Logistic Regression to predict suicidal ideation from online texts. **Multinomial Naive Bayes**analyzes word frequency distributions, **K-Neighbors Classifier** uses nearest data points for classification, and **Logistic Regression** applies a logistic function to estimate probabilities. The selection process involved evaluating each model's accuracy and AUC score, with adjustments for data normalization in optimization models like KNN and LR to enhance performance

## Result  
Results demonstrated the efficacy of various vectorization and optimization models in identifying suicidal ideation from online texts. MultiNB models with TfidVectorizer showed the highest performance and probability of distinguishing depression or suicidal ideation using this model reached 69.18% test accuracy with AUC score 0.77. 

<div>

|  | AUC score  | Train Accuracy  | Test Accuracy  |
| ---------- | -----------| -----------| -----------|
| cvec+ multi_nb  | 0.717 | 0.682| 0.628 |  
| tvec + multi_nb   | 0.754  | 0.687| 0.651| 
|hvec + multi_nb | 0.807 | 0.766| 0.659 | 

</div>
<img src="/image/result.png" width = "500" height = "200" alt="cmt" />

## Application  
The project aims to enhance mental health diagnosis by analyzing suicidal ideation in online texts using advanced NLP techniques. It offers a potential pathway for early intervention and support mechanisms by enabling the timely identification of individuals at risk, thereby contributing to preventative mental health care strategies.
