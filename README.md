# CMPE257-MovieRecommenderSystem
## Project Title: Movie Recommender System

## Team Information:
Ankita Arvind Deshmukh - ankdeshm <br>
Ganesh S Tulshibagwale - Ganesh-S-Tulshibagwale <br>
Indranil Dutta - d1ndra <br>
Pranav Chellagurki - pranav4099 <br>

## About Data:
## Dataset Name: 
Investigating Serendipity in Recommender Systems Based on Real User Feedback
## Source: 
https://grouplens.org/datasets/serendipity-2018/
## Data Summary: 
GroupLens Research group at the University of Minnesota and the University of Jyväskylä conducted an experiment in MovieLens (http://movielens.org) where users were asked how serendipitous particular movies were to them. This dataset contains user answers to GroupLens’ questions and additional information, such as past ratings of these users, recommendations they received before replying to the survey and movie descriptions. The dataset was generated on January 15, 2018. The data are contained in the files ‘answers.csv’, ‘movies.csv’, ‘recommendations.csv’, ‘tag_genome.csv’, ‘tags.csv’ and ‘training.csv’. Overall, there are 10,000,000 ratings (2,150 ratings stored in `answers.csv` and 9,997,850 in ‘training.csv’).

## Problem Description: 
•	To find k-similar users to every user and k-similar items (movies) to every item in the dataset <br>
•	To create user profile and movie profile to identify similarities between these vectors for prediction  <br>
•	To analyze the effect of various movie features such as genres, actors, directors, release date (metadata/content) on the rating prediction <br>
•	To design a model which predicts the ratings for users based on user/item similarity and content <br>
•	To provide movie recommendation to users based on the predicted ratings and perform a qualitative comparison of different approaches <br>

## Potential Methods:
•	Similarity metrics such as Cosine, Raw Cosine, Pearson similarity coefficient etc. <br>
•	User-based collaborative-filtering using similarity among different users <br>
•	Item-based collaborative-filtering using similarity among different items (movies) <br>
•	Content-based recommendation system using feature vector for movies (user-item profile) <br>
•	Latent-matrix factorization-based recommendation system using other metadata <br>





