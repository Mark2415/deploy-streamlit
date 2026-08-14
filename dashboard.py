# Streamlit app prediksi status mahasiswa  - Jaya Jaya Institut

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'model_rf.joblib'
DATA_REF_PATH = BASE_DIR / 'student_data_filtered.csv'

FEATURE_COLS = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Age_at_enrollment',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_1st_sem_enrolled',
    'Admission_grade',
    'Curricular_units_2nd_sem_evaluations',
    'Previous_qualification_grade',
    'Curricular_units_2nd_sem_without_evaluations'
]

@st.dialog('Result')
def show_prediction(output):
    if output == 1:
        st.success('Student Status Prediction: **Graduate**')
    else:
        st.error('Student Status Prediction: **Dropout**')

def data_preprocessing(data_input):
    # Cek separator otomatis (; atau ,)
    with open(DATA_REF_PATH, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
    
    df_ref = pd.read_csv(DATA_REF_PATH, sep=sep)
    if 'Status' in df_ref.columns:
        df_ref = df_ref.drop(columns=['Status'])
    
    df_input_clean = data_input[FEATURE_COLS]
    df_ref_clean = df_ref[FEATURE_COLS]
    
    df_combined = pd.concat([df_input_clean, df_ref_clean])
    df_scaled = StandardScaler().fit_transform(df_combined)
    
 
    return df_scaled[[0]]

def model_predict(df):
    model = joblib.load(MODEL_PATH)
    return model.predict(df)

def main():
    st.title('Jaya Jaya Institute Student Prediction')

    st.subheader('Student Information')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age at Enrollment', min_value=17, max_value=70, value=20, key='age')
    with col2:
        admission_grade = st.number_input('Admission Grade (0-200)', min_value=0, max_value=200, value=100, key='admission_grade')
    with col3:
        prev_qual_grade = st.number_input('Previous Qual. Grade (0-200)', min_value=0, max_value=200, value=100, key='prev_qual_grade')

    st.write('')
    st.markdown('**Semester 1 Performance**')
    col1, col2, col3 = st.columns(3)
    with col1:
        u1_enrolled = st.number_input('Units Enrolled', min_value=0, max_value=26, value=10, key='u1_enrolled')
    with col2:
        u1_approved = st.number_input('Units Approved', min_value=0, max_value=26, value=8, key='u1_approved')
    with col3:
        u1_grade = st.number_input('Average Grade', min_value=0, max_value=20, value=10, key='u1_grade')

    st.write('')
    st.markdown('**Semester 2 Performance**')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        u2_enrolled = st.number_input('Units Enrolled', min_value=0, max_value=23, value=10, key='u2_enrolled')
    with col2:
        u2_approved = st.number_input('Units Approved', min_value=0, max_value=20, value=8, key='u2_approved')
    with col3:
        u2_grade = st.number_input('Average Grade', min_value=0, max_value=20, value=10, key='u2_grade')
    with col4:
        u2_evals = st.number_input('Evaluations Taken', min_value=0, max_value=33, value=8, key='u2_evals')
    
    col1, col2 = st.columns(2)
    with col1:
        u2_no_eval = st.number_input('Units Without Evaluations', min_value=0, max_value=12, value=0, key='u2_no_eval')
    with col2:
        st.write('')
        st.write('')

    # Data harus sesuai urutan FEATURE_COLS
    data = [[
        u2_approved, u2_grade, u1_approved, u1_grade,
        age, u2_enrolled, u1_enrolled,
        admission_grade, u2_evals, prev_qual_grade, u2_no_eval
    ]]

    df_input = pd.DataFrame(data, columns=FEATURE_COLS)

    if st.button('Predict', type='primary'):
        try:
            data_scaled = data_preprocessing(df_input)
            output = model_predict(data_scaled)
            show_prediction(output[0])
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
