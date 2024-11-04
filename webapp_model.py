#webapp modellel
#       python -m streamlit run webapp_model.py
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

if 'model' not in st.session_state:
    try:
        st.session_state.model = joblib.load("random_forest_model.pkl")
        st.session_state.kesz = 1
    except FileNotFoundError:
        st.session_state.kesz = 0


dataset = pd.read_csv("data.csv")

dataset["Gender"] = dataset["Gender"].map({'Female': 1, 'Male': 0})
le = LabelEncoder()
dataset["Interest"] = le.fit_transform(dataset["Interest"])
le_interrest_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

dataset["Personality"] = le.fit_transform(dataset["Personality"])
le_personality_mapping = dict(zip(le.classes_, le.transform(le.classes_)))


y = dataset["Personality"]
X = dataset.drop('Personality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size= 0.8, random_state= 42)


if st.session_state.kesz == 0:
    model = RandomForestClassifier()
    st.session_state.model = model.fit(X_train, y_train)
    joblib.dump(st.session_state.model, "random_forest_model.pkl")
    st.session_state.kesz = 1

#model.fit(X_train, y_train)

model = st.session_state.model
y_pred = model.predict(X_test)

prediction_dict = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

personality_descriptions = {
    'ENFJ': 'ENFJ (The Teacher): Very empathetic, enthusiastic, and supportive people who believe in the power of cooperation and community. They are good leaders and motivators who inspire others to bring out their best.',
    'ENFP': 'ENFP (The Adventurer): Creative, social, and curious individuals who enjoy exploring new opportunities and building connections. They have strong intuition and love bringing new ideas to life.',
    'ENTJ': 'ENTJ (The Leader): Determined and goal-oriented people who see the big picture well and are excellent organizers. They often find themselves in leadership roles and inspire others with strategic thinking.',
    'ENTP': 'ENTP (The Debater): Flexible and inventive individuals who enjoy debating and seeking new ideas. They love facing challenges and are not afraid of the unknown.',
    'ESFJ': 'ESFJ (The Caregiver): Warm-hearted and social people who enjoy helping others and often participate in community work. They value harmony and rules, and they take good care of their surroundings.',
    'ESFP': 'ESFP (The Entertainer): Energetic and spontaneous people who love living in the moment and enjoy being around others. They often find themselves at the center of attention and share their joy with others.',
    'ESTJ': 'ESTJ (The Executive): Practical and organized individuals who like taking control and are excellent problem solvers. They tend to live by the rules and are willing to take on responsibility.',
    'ESTP': 'ESTP (The Entrepreneur): Adventurous and bold people who love practical experience and often seek excitement. They can make quick decisions and adapt well to change.',
    'INFJ': 'INFJ (The Counselor): Intuitive, compassionate, and idealistic people who deeply care about the well-being of others. They have a strong understanding of people and often aim to help others.',
    'INFP': 'INFP (The Idealist): Sensitive and idealistic people who care deeply about their values and the feelings of others. They believe in self-expression and achieving inner harmony.',
    'INTJ': 'INTJ (The Strategist): Analytical and independent thinkers who enjoy developing long-term plans and making strategic decisions. They are determined and not afraid of challenges.',
    'INTP': 'INTP (The Thinker): Logical and curious people who enjoy solving theoretical questions and analyzing complex problems. They often immerse themselves in thinking and seeking new ideas.',
    'ISFJ': 'ISFJ (The Protector): Quiet and caring individuals who are loyal and reliable. They value taking care of others and preserving traditions.',
    'ISFP': 'ISFP (The Artist): Calm and sensitive people who appreciate creativity and beauty. They are inclined toward self-expression and enjoy seeking new experiences.',
    'ISTJ': 'ISTJ (The Inspector): Rational and dependable individuals who like structure and order. They are good problem solvers and pay great attention to detail.',
    'ISTP': 'ISTP (The Craftsman): Practical and independent people who are skilled at working with tools and technology. They like to gain hands-on experience and find solutions.'
}


st.title("Random Forest Classifier webapp")
#st.write(dataset.head())
#st.write(f'Pontosság: {accuracy:.2f}')

#user adatok bekérése:
#st.title('User Feature Input Form')
#st.write(le_interrest_mapping)
#st.write(le_personality_mapping)
age = st.number_input('Age', min_value=0, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
education = st.selectbox('Education', ['No', 'Yes'],help='A binary variable, A value of YES indicates the individual has at least a graduate-level education (or higher), and NO indicates an undergraduate, high school level or Uneducated.')
introversion_score = st.slider('Introversion Score', min_value=0, max_value=10, value=5, help='A continuous variable ranging from 0 to 10, representing the individuals tendency toward introversion versus extraversion. Higher scores indicate a greater tendency toward extraversion.')
sensing_score = st.slider('Sensing Score', min_value=0, max_value=10, value=5,help='A continuous variable ranging from 0 to 10, representing the individuals preference for sensing versus intuition. Higher scores indicate a preference for sensing.')
thinking_score = st.slider('Thinking Score', min_value=0, max_value=10, value=5,help='A continuous variable ranging from 0 to 10, indicating the individuals preference for thinking versus feeling. Higher scores indicate a preference for thinking.')
judging_score = st.slider('Judging Score', min_value=0, max_value=10, value=5,help='A continuous variable ranging from 0 to 10, representing the individuals preference for judging versus perceiving. Higher scores indicate a preference for judging.')
interest = st.selectbox('Interest', ['Unknown', 'Sports', 'Arts', 'Technology', 'Other'],help='A categorical variable representing the individuals primary area of interest.')


#{'Arts': 0, 'Others': 1, 'Sports': 2, 'Technology': 3, 'Unknown': 4}
#{'ENFJ': 0, 'ENFP': 1, 'ENTJ': 2, 'ENTP': 3, 'ESFJ': 4, 'ESFP': 5, 'ESTJ': 6, 'ESTP': 7, 'INFJ': 8, 'INFP': 9, 'INTJ': 10, 'INTP': 11, 'ISFJ': 12, 'ISFP': 13, 'ISTJ': 14, 'ISTP': 15}

#kapott adat átalakítása
gender_num = 1 if gender == 'Male' else 0
education_num = 1 if education == 'Yes' else 0
interest_dict = {'Arts': 0, 'Others': 1, 'Sports': 2, 'Technology': 3, 'Unknown': 4}
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



#ai gondolkodik és kitalálja hogy:
prediction = model.predict(df)

predicted_personality = prediction_dict[prediction[0]]
#st.write(f'Előrejelzett személyiség: **[{predicted_personality}](#{predicted_personality})**')
st.write(f'Előrejelzés: {prediction_dict[prediction[0]]}')

if st.button(f'More detailed description of the {predicted_personality} personality'):
    st.session_state.personality_selected = predicted_personality

if 'personality_selected' in st.session_state:
    st.markdown("### Description:")
    selected_personality = st.session_state.personality_selected
    st.write(personality_descriptions[selected_personality])

#st.write(f'Előrejelzés: {prediction[0]}')
#val = prediction[0]
#st.write(prediction_dict[val])