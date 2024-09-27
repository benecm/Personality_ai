#webapp modellel
#       python -m streamlit run webapp.py
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st


dataset = pd.read_csv("data.csv")

dataset["Gender"] = dataset["Gender"].map({'Female': 1, 'Male': 0})
le = LabelEncoder()
dataset["Interest"] = le.fit_transform(dataset["Interest"])
dataset["Personality"] = le.fit_transform(dataset["Personality"])

y = dataset["Personality"]
X = dataset.drop('Personality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 0.8, random_state= 42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

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