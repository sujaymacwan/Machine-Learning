# Movie Recommendation System

This project develops a complete recommendation system for recommending movies to users, modeled after a simple version of Netflix. Using the MovieLens 100k dataset, it implements content-based filtering, collaborative filtering, and a hybrid approach to deliver personalized movie suggestions. The system leverages Python, Pandas, NumPy, scikit-learn, and the Surprise library to analyze user ratings and movie metadata, achieving high accuracy in recommendations.

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Shooting incidents pose significant challenges to urban safety and resource allocation. This project analyzes shooting data in New York City to identify patterns, hotspots, and trends. By leveraging big data technologies and data visualization tools, the project aims to provide actionable insights to improve public safety measures.

## Objectives
- Build a hybrid movie recommendation system combining content-based and collaborative filtering techniques.
- Personalize movie suggestions for users based on their ratings and preferences.
- Achieve high recommendation accuracy using matrix factorization (SVD) and TF-IDF vectorization.
- Demonstrate scalability and interpretability for real-world application in recommendation engines.

## Dataset
The project uses the MovieLens 100k dataset, collected by the GroupLens Research Project at the University of Minnesota. This dataset includes:

- **100,000 ratings (1-5) from 943 users on 1,682 movies.**
- **Demographic information** of users (age, gender, occupation, zip code).
- **Movie metadata** (titles, genres, release dates, etc.).

### Key Files:
- `u.user`: User demographic data.
- `u.item`: Movie metadata and genres.
- `u.data`: User-movie ratings with timestamps.
- `ua.base` and `ua.test`: Training and testing splits for model evaluation.

## Technologies Used
- **Programming Languages:** Python
- **Libraries:**
  - `pandas`, `numpy`: Data manipulation and analysis.
  - `scikit-learn`: TF-IDF vectorization, Truncated SVD, and feature engineering.
  - `Surprise`: Collaborative filtering with SVD (Singular Value Decomposition).
  - `matplotlib`: Data visualization (optional for exploratory analysis).
- **Data Storage:** CSV files (MovieLens dataset).
- **Development Environment:** Jupyter Notebook, Python 3.x.

## Methodology
The recommendation system implements two main approaches, combined into a hybrid model:

### 1. Content-Based Filtering
- **Data Preparation:** Combined movie titles and genres into a metadata column, using TF-IDF vectorization to represent movie content.
- **Feature Engineering:** Created a `full_metadata` column by concatenating genre-specific metadata (e.g., Action, Adventure) for each movie, enabling content-based similarity calculations.
- **Similarity:** Used cosine similarity on TF-IDF vectors to recommend movies with similar genres and titles to a user’s previously rated movies.

### 2. Collaborative Filtering
- **User-Item Matrix:** Constructed a sparse matrix of user-movie ratings using Pandas pivot tables.
- **Matrix Factorization:** Applied Singular Value Decomposition (SVD) from the Surprise library to predict missing ratings, achieving an RMSE of 0.94 on the test set.
- **Evaluation:** Split data into training (`ua.base`) and testing (`ua.test`) sets to evaluate model performance.

### 3. Hybrid Approach
- Combined content-based and collaborative filtering to recommend movies, prioritizing high-rated, genre-aligned suggestions for users based on their historical ratings.

## Results
- **Recommendation Accuracy:** Achieved an RMSE of **0.94** using SVD, indicating high precision in predicting user ratings.
- **Top Recommendations:** Generated personalized movie lists for users (e.g., User IDs 1, 40, 49, 50, 915), recommending classics like *North by Northwest*, *Shawshank Redemption*, and *Pulp Fiction* based on user preferences.
- **Scalability:** The system scales to handle **943 users** and **1,682 movies**, with potential for larger datasets using distributed computing frameworks.

#### Example recommendations for User ID 1:
```
North by Northwest (1959): 4.73
Third Man, The (1949): 4.60
Schindler's List (1993): 4.55
```

## How to Run
Follow these steps to set up and run the project:

### Prerequisites
- **Software:** Python 3.8 or higher, Jupyter Notebook (optional).
- **Libraries:** Install required dependencies using:
  ```bash
  pip install pandas numpy scikit-learn surprise matplotlib
  ```

### Steps
#### Clone the Repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```
#### Download the Dataset
Download the MovieLens 100k dataset from [here](https://grouplens.org/datasets/movielens/100k/) or use the provided `wget` command in the notebook:
```bash
wget https://raw.githubusercontent.com/subashgandyer/datasets/main/ml-100k/ml-100k.zip
unzip ml-100k.zip
```
#### Run the Notebook
- Open `movie_recommendation.ipynb` in Jupyter Notebook or your preferred Python IDE.
- Execute cells sequentially to load data, build the recommendation system, and generate predictions.

#### Generate Recommendations
- Use the `predict_user_ratings(user_id)` function to recommend movies for any user ID (e.g., `1, 40, 49, 50, 915`).

### Notes
- Ensure all dataset files (`u.user`, `u.item`, `u.data`, `ua.base`, `ua.test`) are in the project directory.
- Adjust the `user_id` in `predict_user_ratings()` to test different users.

## Future Work
- **Real-Time Updates:** Integrate real-time user ratings and streaming data for dynamic recommendations.
- **Scalability:** Use distributed frameworks like PySpark or Dask for larger datasets.
- **Advanced Models:** Explore deep learning-based recommenders (e.g., neural collaborative filtering) or incorporate natural language processing (NLP) for movie descriptions.
- **Evaluation Metrics:** Add precision, recall, and F1-score to complement RMSE for a comprehensive performance assessment.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **Dataset:** MovieLens 100k dataset, provided by the GroupLens Research Project at the University of Minnesota.
- **Inspiration:** Built as a simplified clone of Netflix, inspired by modern recommendation systems in streaming platforms.
- **Contributors:** Sujay Macwan (primary developer).
