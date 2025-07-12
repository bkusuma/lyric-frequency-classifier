# Lyrical Genre Classification

*Can we use the words in song lyrics to accurately predict a song's genre?*  

## About
### Tech Stack
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-Learn, XGBoost, Seaborn, Matplotlib
- **Tools:** Jupyter Notebook, Git, SQLite


Using the Million Song Dataset and several corollaries, I uses ground truth genre classes to build machine learning models to assist with genre classification.


This project is a deep dive into the intersection of Natural Language Processing and music. I developed a machine learning pipeline to classify songs into ten distinct genres based solely on the frequency of words in their lyrics. The result is a robust model that demonstrates the power of lyrical content in defining musical style.


![Distinctive Stemmed Word WordClouds](https://github.com/bkusuma/lyric-frequency-classifier/blob/207524a8f74eb7134b4f8336638e3e82183ba258/assets/4x3_distinctive_words_roboto.png)
*A selection of word clouds showing the most distinctive words for various genres.*

This capstone project successfully demonstrates an end-to-end data science workflow, from data ingestion and cleaning to feature engineering, modeling, and evaluation. The best-performing model, an **XGBoost Classifier**, achieved a promising level of accuracy, proving the viability of using lyric word counts for genre prediction. The **HistGradientBoostingClassifier** fared slightly worse on these metrics but had a much better confusion matrix which showed more promise for a future working overall genre classfier.

A slide deck overview of the project is available here: [Lyrical Genre Classification Presentation](https://docs.google.com/presentation/d/1V0gL-JjEAadghOHZ4bsJmCiABNKccE8COhpDUuzEvZE/edit?usp=share_link)

## The Approach: A Methodical Breakdown

This project followed a structured data science methodology to ensure robust and reproducible results.

### 1. Data Sourcing and Integration
The foundation of this analysis is a combination of three powerful datasets:
- **The Million Song Dataset (MSD):** Provided the core track metadata.
- **The musiXmatch Dataset:** The official lyrics collection for the MSD, containing a bag-of-words representation for over 200,000 tracks.
- **The Top-MAGD Dataset:** Used to map tracks to their top-level musical genres.

These datasets were meticulously cleaned, merged, and processed to create a unified dataset ready for analysis.

### 2. Exploratory Data Analysis (EDA)
Before modeling, I conducted a thorough EDA to understand the data's characteristics and uncover initial insights. This included:
- **Genre Distribution Analysis:** Visualizing the frequency of each of the 10 genres in the dataset to understand class balance.
- **Distinctive Word Analysis:** Generating word clouds for each genre to visually identify the lyrical features that make each style unique. This step was crucial for confirming that lyrical content held a strong enough signal for classification.

### 3. Modeling and Evaluation
The core of the project was to build and compare multiple classification models to find the most effective approach.

- **Feature Set:** The primary features were the top 5,000 most frequent words from the musiXmatch dataset, forming a classic Bag-of-Words model. Additional features like song duration and release year were also included.
- **Models Tested:** I implemented and evaluated a suite of powerful classification algorithms:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbors
    - Gaussian Naive-Bayes
    - **HistGradientBoostingClassifier**
    - **XGBoost Classifier**

- **Performance:** The models were evaluated on their accuracy and F1-score. The tree-based ensemble methods demonstrated the strongest performance.

| Model                             | Test Accuracy | Test F1-Score |
| --------------------------------- | ------------- | ------------- |
| **XGBoost** | **42.5%** | **40.3%** |
| **HistGradientBoostingClassifier**| **41.0%** | **40.0%** |
| Logistic Regression               | 37.0%         | 35.5%         |
| Random Forest                     | 38.3%         | 32.2%         |



![XGBoost Confusion Matrix](https://github.com/bkusuma/lyric-frequency-classifier/blob/207524a8f74eb7134b4f8336638e3e82183ba258/assets/confusion_xgb.png?raw=true)

*The confusion matrix for the XGBoost model.*


![HistGradientBoostingClassifier Confusion Matrix](https://github.com/bkusuma/lyric-frequency-classifier/blob/207524a8f74eb7134b4f8336638e3e82183ba258/assets/confusion_hgb.png?raw=true)

*The confusion matrix for the HistGradientBoostingClassifier model.*

## Licensing

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Citations

1. Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.
