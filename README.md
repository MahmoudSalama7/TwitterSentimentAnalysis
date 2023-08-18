![image](https://github.com/MahmoudSalama7/TwitterSentimentAnalysis/assets/104441438/aa4a7c71-2ad5-46a7-824a-7f4bceafc036)

# Twitter Sentiment Analysis with Bag-of-Words (BoW) Model

This repository contains a sentiment analysis model built using a Bag-of-Words (BoW) approach for analyzing Twitter tweets. The model utilizes Support Vector Machine (SVM) classification to predict sentiment labels based on the text content of the tweets. The dataset used for training and evaluation consists of 1.6 million rows and 6 columns. Due to compilation time constraints, a subset of 1000 samples was used for model development.

## Dataset
The original dataset contains 1.6 million rows and 6 columns, which include various attributes of the Twitter tweets. The sentiment labels are the target variable for the sentiment analysis task. However, to expedite the model development process, a representative sample of 1000 tweets was selected from the original dataset.

## Usage
1. **Dataset Preparation**: The original dataset is too large for quick model development. To get started, use the provided script `data_sampling.py` to create a representative sample of 1000 tweets from the original dataset. The sampled data is stored in a CSV file named `sampled_tweets.csv`.

2. **Data Preprocessing**: The `preprocess.py` script takes care of data cleaning and preprocessing tasks such as removing special characters, tokenization, and creating the Bag-of-Words representation. The processed data is saved in a file named `preprocessed_data.csv`.

3. **Model Training**: The main model is trained using the processed data. The `train_model.py` script reads the preprocessed data, constructs the BoW features, and trains an SVM classifier. The trained model is then saved as `sentiment_model.pkl`.

4. **Model Evaluation**: To evaluate the model's performance, the `evaluate_model.py` script loads the trained SVM classifier and evaluates it on the test data. The accuracy score and other relevant metrics are displayed in the console.

## Requirements
- Python 3.x
- pandas
- scikit-learn

## Usage Example
1. Run `data_sampling.py` to generate the sampled dataset.
2. Execute `preprocess.py` to preprocess the data and create the BoW representation.
3. Train the sentiment analysis model by running `train_model.py`.
4. Evaluate the trained model using `evaluate_model.py`.

## Results
The model achieved an accuracy of approximately 0.66 on the test dataset. Note that this accuracy result is based on a relatively small subset of the original dataset and further tuning and experimentation could potentially lead to improved results.

## Future Improvements
1. Experiment with different text preprocessing techniques to enhance feature representation.
2. Explore more advanced machine learning models and algorithms for sentiment analysis.
3. Consider using a larger representative sample or utilizing techniques like cross-validation for more reliable model evaluation.

## Disclaimer
Please note that this project serves as a basic example of sentiment analysis using a Bag-of-Words approach. The accuracy achieved is not state-of-the-art and is meant for educational and illustrative purposes.

Feel free to contribute, enhance, and adapt the code according to your requirements. If you have any questions or suggestions, please open an issue in the repository.

**Happy Coding!**
