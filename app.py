import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import re

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
INGREDIENTS_PATH = BASE_DIR / "list_ingredients_for_app.csv"
FEATURES_PATH = BASE_DIR / "concatenated_features.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
RECIPES_PATH = BASE_DIR / "data" / "final_ingredient_list_created.csv"

st.title("Let us help you get ideas for your next meal")


@st.cache_data
def load_ingredients():
    return pd.read_csv(INGREDIENTS_PATH)["0"].dropna().unique()


@st.cache_data
def load_recipes():
    return pd.read_csv(RECIPES_PATH)


@st.cache_resource
def load_model_artifacts():
    with open(FEATURES_PATH, "rb") as file:
        features = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return features, vectorizer


def user_input_transformer(input_string):
    return " ".join(input_string.lower().split())

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
ingredient_list = load_ingredients()

user_ingredients = st.multiselect(
    'What is in your fridge',
    ingredient_list)

st.write('You selected:', user_ingredients)
#whats_in_your_fridge = "eggs mushroom chilli rice noodles"
user_input = str(user_cuisine)+ ' ' + str(user_preference) + ' ' + str(user_course)+ ' ' + str(user_ingredients)
preprocessed_input = user_input_transformer(user_input)

#st.button("Reset", type="primary")
if st.button('Generate Recipes'):

    X_loaded, vectorizer = load_model_artifacts()

    # TF-IDF Vectorization of user input
    user_input_vector = vectorizer.transform([preprocessed_input])

    # Calculate cosine similarity between user input and recipes
    similarity_scores = cosine_similarity(user_input_vector, X_loaded)

    # Get indices of top recommended recipes (top 3 in this example)
    top_indices = similarity_scores.argsort()[0][-3:][::-1]

    df = load_recipes()

    # Display top recommended recipes
    st.header("\nTop Recommended Recipes:")
    for idx in top_indices:
        st.subheader("Recipe")
        st.write(df.loc[idx, 'TranslatedRecipeName'].split(' - ')[0])
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
            st.subheader("Ingredients")
            st.write(str(df.loc[idx,'FinalIngredientList']))
            st.subheader("How to prepare")
            input_string =df.loc[idx, 'TranslatedInstructions']
            # Split the input string into sentences using regex
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s+|\s*$)', input_string)
            #st.write(('\n').join(df.loc[idx, 'TranslatedInstructions'].split('.'))
            sentences = [sentence for sentence in sentences if sentence not in [' ','']]
            # Convert sentences to Markdown format
            markdown_output = '\n'.join(f"- {sentence}" for sentence in sentences)

            # Print the Markdown output
            st.markdown(markdown_output)
        
