# Movie Recommendation System with Hybrid ML Approach

This project demonstrates the implementation of a hybrid movie recommendation system using classical machine learning models and semantic embeddings. It is based on the MovieLens 100k dataset and developed entirely in a Jupyter Notebook using Google Colab.

## Dataset
We use the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/), which contains 100,000 ratings from 943 users on 1682 movies. The dataset includes:
- User-item interactions (ratings)
- Movie titles and genres

## Methods Used
The project follows a step-by-step approach:
1. **Data Preparation**: Merging user ratings with movie metadata (titles and genres).
2. **Feature Engineering**:
   - Encoded `user_id` and `item_id`
   - One-hot encoded genres (19 binary features)
   - Text-based movie title embeddings using **Sentence-BERT**
3. **Modeling**:
   - Trained and compared **Random Forest** and **Gradient Boosting Regressor**
   - Evaluated using RMSE
   - Best performance: Gradient Boosting + SBERT + genres
4. **Recommendation Logic**:
   - For a given user, predict ratings for unseen movies
   - Sort and recommend top-N movies based on predicted scores

## Technologies
- Python
- Pandas, Scikit-learn
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- Google Colab

## Sample Results
- **Random Forest RMSE**: ~1.115
- **Gradient Boosting RMSE**: ~1.047
- Model performance improved after including genre and semantic title embeddings


## Google Drive Project Folder
The project including the dataset and notebook is available here:  
[ Google Drive Link]([https://drive.google.com/drive/folders/1my3OYdsaxDCcyOwQr5DvgozO2ApArOAG?usp=sharing](https://drive.google.com/drive/folders/1my3OYdsaxDCcyOwQr5DvgozO2ApArOAG?usp=sharing))

## Future Improvements
- Add user profile features (age, gender)
- Include collaborative filtering matrix factorization
- Build a Streamlit or FastAPI interface

---

Feel free to use or modify this notebook for your own projects. Star ⭐ this repo if you find it useful!

---

