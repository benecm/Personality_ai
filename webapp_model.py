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
    'ENFJ': 'ENFJ (A Tanító): Nagyon empatikus, lelkes és támogató személyek, akik hisznek az együttműködés és a közösség erejében. Jó vezetők és motivátorok, akik másokat is arra inspirálnak, hogy kihozzák magukból a legjobbat.',
    'ENFP': 'ENFP (A Kalandor): Kreatív, szociális és kíváncsi személyek, akik élvezik az új lehetőségek felfedezését és az emberek közötti kapcsolatok kialakítását. Nagyon jó intuícióval rendelkeznek, és szeretnek új ötleteket megvalósítani.',
    'ENTJ': 'ENTJ (A Vezető): Határozott és céltudatos emberek, akik jól átlátják a nagyobb képet és kiváló szervezők. Gyakran vezető szerepben találják magukat, és képesek stratégiai gondolkodással másokat is inspirálni.',
    'ENTP': 'ENTP (A Vita Kedvelő): Rugalmas és találékony személyek, akik szívesen vitatkoznak és új ötleteket keresnek. Szeretnek kihívásokkal szembenézni, és nem riadnak vissza az ismeretlentől.',
    'ESFJ': 'ESFJ (A Gondoskodó): Melegszívű és szociális emberek, akik szívesen segítenek másokon és gyakran vesznek részt közösségi munkában. Értékelik a harmóniát és a szabályokat, és jól gondoskodnak a környezetükről.',
    'ESFP': 'ESFP (A Szórakoztató): Energetikus és spontán személyek, akik szeretnek a jelenben élni és élvezik a társaságot. Gyakran a társaság középpontjában állnak, és örömmel osztják meg örömüket másokkal.',
    'ESTJ': 'ESTJ (A Végrehajtó): Gyakorlatias és szervezett személyek, akik szeretik az irányítást és kiváló problémamegoldók. Hajlamosak szabályok szerint élni, és szívesen vállalják a felelősséget.',
    'ESTP': 'ESTP (A Vállalkozó): Kalandvágyó és bátor emberek, akik szeretnek a gyakorlatban tapasztalni és gyakran keresik az izgalmakat. Képesek gyors döntéseket hozni és jól alkalmazkodnak a változáshoz.',
    'INFJ': 'INFJ (A Tanácsadó): Intuitív, együttérző és idealista személyek, akik mélyen törődnek mások jólétével. Nagyon jó meglátásaik vannak az emberekkel kapcsolatban, és gyakran céljuk mások segítése.',
    'INFP': 'INFP (Az Idealiszta): Érzékeny és idealista emberek, akik mélyen törődnek értékeikkel és mások érzéseivel. Hisznek az önkifejezésben és a belső harmónia elérésében.',
    'INTJ': 'INTJ (A Stratéga): Analitikus és független gondolkodók, akik szeretnek hosszú távú terveket kidolgozni és stratégiai döntéseket hozni. Határozottak, és nem riadnak vissza a kihívásoktól.',
    'INTP': 'INTP (A Gondolkodó): Logikus és kíváncsi emberek, akik élvezik az elméleti kérdések megoldását és az összetett problémák elemzését. Gyakran mélyen elmerülnek a gondolkodásban és az új ötletek keresésében.',
    'ISFJ': 'ISFJ (A Védelmező): Csendes és gondoskodó személyek, akik hűségesek és megbízhatóak. Fontos számukra a másokról való gondoskodás és a hagyományok megőrzése.',
    'ISFP': 'ISFP (A Művész): Nyugodt és érzékeny emberek, akik szeretik a kreativitást és a szépséget. Hajlamosak az önkifejezésre és szeretnek új élményeket keresni.',
    'ISTJ': 'ISTJ (A Megfigyelő): Racionális és megbízható személyek, akik szeretik a struktúrát és a rendszert. Jó problémamegoldók és kiváló figyelmet fordítanak a részletekre.',
    'ISTP': 'ISTP (A Mesterember): Gyakorlati és független emberek, akik jól dolgoznak eszközökkel és technológiával. Szeretnek közvetlen tapasztalatokat szerezni és megoldásokat találni.'
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
st.write(f'Előrejelzés: {prediction[0]}')

if st.button(f'Részletes leírás {predicted_personality} személyiségről'):
    st.session_state.personality_selected = predicted_personality

if 'personality_selected' in st.session_state:
    st.markdown("### Személyiség leírás")
    selected_personality = st.session_state.personality_selected
    st.write(personality_descriptions[selected_personality])

#st.write(f'Előrejelzés: {prediction[0]}')
#val = prediction[0]
#st.write(prediction_dict[val])