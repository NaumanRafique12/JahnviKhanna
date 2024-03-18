import streamlit as st
import pandas as pd

import joblib
import numpy as np
import sklearn
# Specify the path to your pickle file
pickle_file_path = 'pipeLR.pkl'
# @st.cache(allow_output_mutation=True)
# Open the pickle file for reading in binary mode
def load_Model():
    with open(pickle_file_path, 'rb') as f:
        # Load the data from the pickle file
        pipe = joblib.load(f)
    return pipe
# Now you can use the loaded data, which could be a dictionary, list, etc.
pipe = load_Model()
st.title('Indian House Price Prediction')
st.header('By: Jahnvi Khanna')


# Gather input from the user
number_of_bedrooms = st.number_input('Number of Bedrooms', min_value=0)
number_of_bathrooms = st.number_input('Number of Bathrooms', min_value=0)
living_area = st.number_input('Living Area (sqft)', min_value=0)
lot_area = st.number_input('Lot Area (sqft)', min_value=0)
number_of_floors = st.number_input('Number of Floors', min_value=0)
waterfront_present = st.checkbox('Waterfront Present')
number_of_views = st.number_input('Number of Views', min_value=0)
condition_of_the_house = st.selectbox('Condition of the House', [1, 2,3,4,5])
grade_of_the_house = st.selectbox('Grade of the House', [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
area_of_the_house_excluding_basement = st.number_input('Area of the House (excluding basement)', min_value=0)
area_of_the_basement = st.number_input('Area of the Basement', min_value=0)
built_year = st.number_input('Built Year', min_value=0)
postal_code = st.number_input('Postal Code')
latitude = st.number_input('Latitude')
living_area_renov = st.number_input('Living Area after Renovation', min_value=0)
lot_area_renov = st.number_input('Lot Area after Renovation', min_value=0)
number_of_schools_nearby = st.number_input('Number of Schools Nearby', min_value=0)
distance_from_the_airport = st.number_input('Distance from the Airport (miles)', min_value=0)

data = {
    'number of bedrooms': [number_of_bedrooms],
    'number of bathrooms': [number_of_bathrooms],
    'living area': [living_area],
    'lot area': [lot_area],
    'number of floors': [number_of_floors],
    'waterfront present': [waterfront_present],
    'number of views': [number_of_views],
    'condition of the house': [condition_of_the_house],
    'grade of the house': [grade_of_the_house],
    'Area of the house(excluding basement)': [area_of_the_house_excluding_basement],
    'Area of the basement': [area_of_the_basement],
    'Built Year': [built_year],
    'Postal Code': [postal_code],
    'Lattitude': [latitude],
    'living_area_renov': [living_area_renov],
    'lot_area_renov': [lot_area_renov],
    'Number of schools nearby': [number_of_schools_nearby],
    'Distance from the airport': [distance_from_the_airport]
}

btn = st.button("Submit")
if btn:
    # Create DataFrame
    df = pd.DataFrame(data)
    st.write(f"Price of this house is {np.exp(pipe.predict(df))[0]}K INR")