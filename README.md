# 🎵 Song Recommendation System

Welcome to the **Song Recommendation System** project! This system is designed to provide personalized song recommendations to users by analyzing their listening behavior and predicting songs they may enjoy. This README provides a comprehensive overview of the project, including installation, methodology, evaluation, and usage instructions.

## 🚀 Project Overview

The goal of this project is to develop a Song Recommendation System that can recommend songs based on historical user data. The model will generate a list of 10 recommended songs for each user from the `recommendation_val.csv` file. This project includes data cleaning, feature engineering, model training, and model evaluation to ensure the best recommendations.

## 📂 Project Structure

- **data/**: Contains datasets used for training and validation, including `recommendation_val.csv`.
- **notebooks/**: Jupyter notebooks used for data exploration, training, and evaluation.
- **src/**: Python scripts for data processing, model training, and evaluation.
- **README.md**: Project overview and instructions.

## 🛠️ Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/GopalMarmat/song-recommendation-system.git
cd song-recommendation-system
pip install -r requirements.txt
```

## 📈 Methodology

1. **Data Visualization**: Visualize the data by scatter plot and detect outlier.
2. **Data Cleaning**: Pre-process the data by handling missing values, removing duplicates, remove outliers, and transforming data types to ensure consistency.
3. **Model Selection**: Implement the Apriori Algorithm to generate association rules and identify song recommendations based on users’ past listening behavior.
4. **Evaluation**: Evaluate the model’s performance by generating 10 song recommendations for each user in the validation dataset.

## 🔍 Evaluation Metric

The model will be evaluated based on its ability to recommend 10 relevant songs for each user ID in `recommendation_val.csv`. Evaluation metrics such as **precision**, **recall**, and **F1-score** are used to measure the relevance of the recommendations.

## 📜 Usage

### 1. Data Preparation
Make sure your training data is available in the `data` folder and is pre-processed. Use the provided scripts or notebooks to clean and prepare the data.

```bash
python src/data_cleaning.py
```

### 2. Model Training and Generating Recommendations
Use the following command to train the model:

```bash
python src/recommendation_model.py
```

This will train the recommendation model using the Apriori algorithm and save the trained model for future predictions.

 --input data/recommendation_val.csv --output data/recommendations.csv

## 📊 Results

The output file `recommendations.csv` will contain the top 10 recommended songs for each user.

The table above csv an example of the output recommendations generated by the model.

## 🔗 Future Improvements

- **Incorporate Collaborative Filtering**: Use collaborative filtering techniques to improve recommendation quality by understanding user similarity.
- **Include Deep Learning Models**: Explore deep learning-based approaches, such as matrix factorization or neural collaborative filtering, for enhanced predictions.
- **Add Real-Time Recommendation**: Implement real-time recommendation generation for a dynamic and interactive user experience.

## 🤝 Contributing

We welcome contributions! Feel free to submit a pull request or open an issue to propose any improvements.
