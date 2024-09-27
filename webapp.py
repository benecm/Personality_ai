#webapp
#futtatas:   python -m streamlit run webapp.py
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib



path = r'C:\Users\majer\Desktop\Perosnality AI\model.pkl'

model = joblib.load(path)

st.title("Random Forest Classifier webapp")
#st.write(dataset.head())
#st.write(f'Pontosság: {accuracy:.2f}')

#user adatok bekérése:
st.title('User Feature Input Form')

age = st.number_input('Age', min_value=0, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
education = st.selectbox('Education', ['No', 'Yes'])
introversion_score = st.slider('Introversion Score', min_value=0, max_value=10, value=5)
sensing_score = st.slider('Sensing Score', min_value=0, max_value=10, value=5)
thinking_score = st.slider('Thinking Score', min_value=0, max_value=10, value=5)
judging_score = st.slider('Judging Score', min_value=0, max_value=10, value=5)
interest = st.selectbox('Interest', ['Unknown', 'Sports', 'Arts', 'Technology', 'Other'])

#kapott adat átalakítása
gender_num = 1 if gender == 'Male' else 0
education_num = 1 if education == 'Yes' else 0
interest_dict = {'Unknown': 0, 'Sports': 1, 'Arts': 2, 'Technology': 3, 'Other': 4}
interest_num = interest_dict[interest]

data = {
    'Age': [age],
    'Gender': [gender_num],
    'Education': [education_num],
    'Introversion Score': [introversion_score],
    'Sensing Score': [sensing_score],
    'Thinking Score': [thinking_score],
    'Judging Score': [judging_score],
    'Interest': [interest_num]
}
df = pd.DataFrame(data)

st.write('DataFrame:')

#ai gondolkodik és kitalálja hogy:
prediction = model.predict(df)
st.write(f'Előrejelzés: {prediction[0]}')