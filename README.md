# **Overview**

RECIPE RECOMENDATIONS - project aims to enhance the culinary experience by providing users with personalized suggestions for recipes similar to those they have shown interest in. Leveraging advanced natural language processing techniques and machine learning, this system recommends recipes that share textual similarities with the user's preferred recipes. 

# **Example**

On this gif, an example of searching for similar recipes to the given [recipe of an apple pie](https://www.allrecipes.com/recipe/16268/apple-pie/) is provided.
![demo_apple_pie](https://github.com/dashayvs/recipe_recomendation/assets/101887992/b5c49868-692e-410f-aa04-fe3c0fcbab53)

The project's raw data was collected by parsing recipe information from the popular recipe-sharing website, Allrecipes. This includes details such as ingredients, instructions, titles, etc.

During the development of the project, 4 models were created to compare and select the best 
*(models.py)*

*WordsComparison Model:*

Uses NLTK for lemmatization and stop words removal.
Computes the number of matching unique words between a query object and each row in the dataset.
Returns indices of rows with the highest number of matching words.

*TfidfSimilarity Model:*

Combines word-level and character-level TF-IDF representations.
Utilizes cosine similarity to measure the likeness between a query object and the entire dataset.
Returns indices of rows with the highest similarity scores.

*ObjectsTextSimilarity Model:*

Employs Sentence Transformer for text embedding.
Concatenates embeddings of different text features to create a representation for each object.
Utilizes cosine similarity to find the most similar objects to a query.

*ObjectsSimilarityFiltered Model:*

Similar to ObjectsTextSimilarity but with an additional filtering step.
Filters similar objects based on specific features and their values.


Based on the validation results, it was concluded that the best model - ObjectsTextSimilarity Model

For the convenience of using this project, an api (app.py) was developed, where the user enters information, can filter if necessary and receives recipes similar to the request

