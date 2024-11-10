import streamlit as st
import pickle
import pandas as pd
import math

# Teams and cities options
teams = ['Mumbai Indians', 'Kolkata Knight Riders', 'Rajasthan Royals', 'Chennai Super Kings', 'Sunrisers Hyderabad',
         'Delhi Capitals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans', 'Royal Challengers Bengaluru']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur', 'Hyderabad', 'Chennai', 'Cape Town',
          'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Indore', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Rajkot', 'Kanpur', 'Bengaluru', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow',
          'Guwahati', 'Mohali']

# Load the machine learning model (pipe.pkl)
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("The model file (pipe.pkl) is missing. Please ensure the file is in the repository.")
    st.stop()  # Stop further execution if model is missing
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Streamlit app title
st.title('IPL Win Predictor')

# Create input fields for user to select teams and city, and input target and match details
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)

with col4:
    overs = st.number_input('Overs completed', min_value=0.0, format="%.1f")

with col5:
    wickets = st.number_input('Wickets out', min_value=0)

# Initialize the variables for prediction
runs_left = balls_left = wickets = crr = rrr = 0

# Prediction button
if st.button('Predict Probability'):
    # Calculate match parameters
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Prepare the input DataFrame for prediction
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    input_df.fillna(0, inplace=True)

    # Try to make predictions with the model
    try:
        result = pipe.predict_proba(input_df)

        # Extract the probabilities
        loss = result[0][0]
        win = result[0][1]

        # Display the results
        st.header(f"{batting_team} - {round(win * 100, 2)}%")
        st.header(f"{bowling_team} - {round(loss * 100, 2)}%")

    except ValueError as e:
        st.error(f"An error occurred during prediction: {e}")

