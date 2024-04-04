import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.metrics.pairwise import cosine_similarity

st.title("Let us help you get ideas for your next meal")

def user_input_transformer(input_string):
    clean_output = ('').join(input_string.lower().split('  '))
    return clean_output

#user_input_transformer(user_input)

# Preprocess user input

# Get cuisine
cuisine_list = ['Indian', 'South Indian Recipes', 'Andhra', 'Udupi', 'Mexican',
       'Fusion', 'Continental', 'Bengali Recipes', 'Punjabi', 'Chettinad',
       'Tamil Nadu', 'Maharashtrian Recipes', 'North Indian Recipes',
       'Italian Recipes', 'Sindhi', 'Thai', 'Chinese',
       'Gujarati Recipes', 'Coorg', 'Rajasthani', 'Asian',
       'Middle Eastern', 'Coastal Karnataka', 'European',
       'Kerala Recipes', 'Kashmiri', 'Karnataka', 'Lucknowi',
       'Hyderabadi', 'Side Dish', 'Goan Recipes', 'Arab', 'Assamese',
       'Bihari', 'Malabar', 'Himachal', 'Awadhi', 'Cantonese',
       'North East India Recipes', 'Sichuan', 'Mughlai', 'Japanese',
       'Mangalorean', 'Vietnamese', 'British', 'Parsi Recipes', 'Greek',
       'Nepalese', 'Oriya Recipes', 'French', 'Indo Chinese', 'Konkan',
       'Mediterranean', 'Sri Lankan', 'Uttar Pradesh', 'Malvani',
       'Indonesian', 'African', 'Shandong', 'Korean', 'American',
       'Kongunadu', 'Pakistani', 'Caribbean', 'North Karnataka',
       'South Karnataka', 'Haryana', 'Appetizer',
       'Uttarakhand-North Kumaon', 'World Breakfast', 'Malaysian',
       'Dessert', 'Hunan', 'Dinner', 'Jewish', 'Burmese',
       'Afghan', 'Jharkhand', 'Nagaland' ]

user_cuisine = st.selectbox(
    'Which cuisine would you like to explore today',
    tuple(cuisine_list))

st.write('You selected:', user_cuisine)

### Get preference 

preference_list = ['Diabetic Friendly', 'Vegetarian', 'High Protein Vegetarian',
       'Non Vegeterian', 'High Protein Non Vegetarian', 'Eggetarian',
       'No Onion No Garlic (Sattvic)', 'Gluten Free', 'Vegan',
       'Sugar Free Diet']
user_preference = st.selectbox(
    'Do you have any dietary restrictions/preferences?',
    tuple(preference_list))
st.write('You selected:', user_preference)

####Get meal type 
course_list = ['Side Dish', 'Main Course', 'Breakfast', 'Lunch',
       'Snack', 'Dinner', 'Appetizer','Dessert', 'North Indian Breakfast',
       'One Pot Dish', 'Brunch', 'Vegan']

user_course = st.selectbox(
    'Which meal of the day do you need suggestions for?',
    tuple(course_list))

### Get user ingredients 
ingredient_list = pd.read_csv('list_ingredients_for_app.csv')['0'].unique()

user_ingredients = st.multiselect(
    'What is in your fridge',
    ingredient_list)

st.write('You selected:', user_ingredients)
#whats_in_your_fridge = "eggs mushroom chilli rice noodles"
user_input = str(user_cuisine)+ ' ' + str(user_preference) + ' ' + str(user_course)+ ' ' + str(user_ingredients)
preprocessed_input = user_input_transformer(user_input)

#st.button("Reset", type="primary")
if st.button('Generate Recipes'):

    x_file_path = "concatenated_features.pkl"
    # To load X from the pickle file
    with open(x_file_path, 'rb') as file:
        X_loaded = pickle.load(file)

    # TF-IDF Vectorization
    vector_file_path = "vectorizer.pkl"
    # To load Vectorizer from the pickle file
    with open(vector_file_path, 'rb') as file:
        vectorizer = pickle.load(file)

    # TF-IDF Vectorization of user input
    user_input_vector = vectorizer.transform([preprocessed_input])

    # Calculate cosine similarity between user input and recipes
    similarity_scores = cosine_similarity(user_input_vector, X_loaded)

    # Get indices of top recommended recipes (top 3 in this example)
    top_indices = similarity_scores.argsort()[0][-3:][::-1]

    df = pd.read_csv("data/final_ingredient_list_created.csv")

    # Display top recommended recipes
    st.header("\nTop Recommended Recipes:")
    for idx in top_indices:
        st.subheader("Recipe")
        st.write(df.loc[idx, 'TranslatedRecipeName'])
        with st.expander("Learn More"):
            cuisine_str = "Cuisine:" + str(df.loc[idx, 'Cuisine'])
            st.subheader("Cuisine:")
            st.write(str(df.loc[idx, 'Cuisine']))
            st.subheader("Course:")
            st.write(df.loc[idx, 'Course'])
            st.subheader("Match Score:")
            st.write(str(round(similarity_scores[0][idx]*100))+"%")
            st.subheader("Diet Type")
            st.write(df.loc[idx, 'Diet'])
            st.subheader("How to prepare")
            input_string =df.loc[idx, 'TranslatedInstructions']
            # Split the input string into sentences using regex
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s+|\s*$)', input_string)
            #st.write(('\n').join(df.loc[idx, 'TranslatedInstructions'].split('.'))
            sentences = [sentence for sentence in sentences if sentence not in [' ','']]
            print(sentences)
            # Convert sentences to Markdown format
            markdown_output = '\n'.join(f"- {sentence}" for sentence in sentences)

            # Print the Markdown output
            st.write(markdown_output)
        