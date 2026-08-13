# Streamlit app untuk prediksi status mahasiswa (Graduate/Dropout) menggunakan model Random Forest - Jaya Jaya Institut

import io
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

buffer = io.BytesIO()

GENDER_MAP = {'Male': 1, 'Female': 0}

MARITAL_MAP = {
    'Single': 1, 'Married': 2, 'Widower': 3,
    'Divorced': 4, 'Facto Union': 5, 'Legally Seperated': 6
}

APPLICATION_MAP = {
    '1st Phase - General Contingent': 1,
    '1st Phase - Special Contingent (Azores Island)': 5,
    '1st Phase - Special Contingent (Madeira Island)': 16,
    '2nd Phase - General Contingent': 17,
    '3rd Phase - General Contingent': 18,
    'Ordinance No. 612/93': 2,
    'Ordinance No. 854-B/99': 10,
    'Ordinance No. 533-A/99, Item B2 (Different Plan)': 26,
    'Ordinance No. 533-A/99, Item B3 (Other Institution)': 27,
    'International Student (Bachelor)': 15,
    'Over 23 Years Old': 39,
    'Transfer': 42,
    'Change of Course': 43,
    'Holders of Other Higher Courses': 7,
    'Short Cycle Diploma Holders': 53,
    'Technological Specialization Diploma Holders': 44,
    'Change of Institution/Course': 51,
    'Change of Institution/Course (International)': 57,
}

FEATURE_COLS = [
    'Marital_status', 'Application_mode', 'Previous_qualification_grade',
    'Admission_grade', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'Age_at_enrollment',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
]

def data_preprocessing(data_input, single_data, n):
    df_ref = pd.read_csv('student_data_filtered.csv').drop(columns=['Status'])
    df = pd.concat([data_input, df_ref])
    df = StandardScaler().fit_transform(df)
    return df[[n]] if single_data else df[0:n]

def model_predict(df):
    model = joblib.load('model_rf.joblib')
    return model.predict(df)

def color_mapping(value):
    color = 'green' if value == 'Graduate' else 'red'
    return f'color: {color}'

@st.dialog('Result')
def show_prediction(output):
    if output == 1:
        st.success('Student Status Prediction: **Graduate**')
    else:
        st.error('Student Status Prediction: **Dropout**')

def main():
    st.title('Jaya Jaya Institute Student Prediction')

    tab_single, tab_multiple = st.tabs(['Single Data', 'Multiple Data'])

    with tab_single:
        col_gender, col_age, col_marital = st.columns([2, 2, 3])
        with col_gender:
            gender = st.radio('Gender', ['Male', 'Female'])
        with col_age:
            age = st.number_input('Age at Enrollment', min_value=17, max_value=70)
        with col_marital:
            marital_status = st.selectbox('Marital Status', list(MARITAL_MAP.keys()))

        st.write('')

        col_app, col_prev, col_adm = st.columns([3, 1.65, 1.1])
        with col_app:
            application_mode = st.selectbox('Application Mode', list(APPLICATION_MAP.keys()))
        with col_prev:
            prev_grade = st.number_input('Previous Qualification Grade', min_value=0, max_value=200)
        with col_adm:
            admission_grade = st.number_input('Admission Grade', min_value=0, max_value=200)

        col_s, col_t, col_d, col_db = st.columns([1.7, 2.1, 1.55, 1])
        with col_s:
            scholarship = 1 if st.checkbox('Scholarship') else 0
        with col_t:
            tuition = 1 if st.checkbox('Tuition up to date') else 0
        with col_d:
            displaced = 1 if st.checkbox('Displaced') else 0
        with col_db:
            debtor = 1 if st.checkbox('Debtor') else 0

        st.write('')

        col_1e, col_2e, col_2ev = st.columns([1, 1, 1.2])
        with col_1e:
            u1_enrolled = st.number_input('Units 1st Sem Enrolled', min_value=0, max_value=26)
        with col_2e:
            u2_enrolled = st.number_input('Units 2nd Sem Enrolled', min_value=0, max_value=23)
        with col_2ev:
            u2_eval = st.number_input('Units 2nd Sem Evaluations', min_value=0, max_value=33)

        col_1a, col_2a, col_2n = st.columns([1, 1, 1.2])
        with col_1a:
            u1_approved = st.number_input('Units 1st Sem Approved', min_value=0, max_value=26)
        with col_2a:
            u2_approved = st.number_input('Units 2nd Sem Approved', min_value=0, max_value=20)
        with col_2n:
            u2_noeval = st.number_input('Units 2nd Sem No Evaluations', min_value=0, max_value=12)

        col_1g, col_2g, _ = st.columns([1, 1, 1.2])
        with col_1g:
            u1_grade = st.number_input('Units 1st Sem Grade', min_value=0, max_value=20)
        with col_2g:
            u2_grade = st.number_input('Units 2nd Sem Grade', min_value=0, max_value=20)

        data = [[
            MARITAL_MAP[marital_status], APPLICATION_MAP[application_mode],
            prev_grade, admission_grade, displaced, debtor, tuition,
            GENDER_MAP[gender], scholarship, age,
            u1_enrolled, u1_approved, u1_grade,
            u2_enrolled, u2_eval, u2_approved, u2_grade, u2_noeval
        ]]

        df = pd.DataFrame(data, columns=FEATURE_COLS)

        if st.button('Predict'):
            data_input = data_preprocessing(df, True, 0)
            output = model_predict(data_input)
            show_prediction(output)

    with tab_multiple:
        with st.expander('**User Guide**'):
            st.write("""
                1. Download the student data Excel template.
                2. Fill in all student data columns.
                3. Upload the completed Excel file.
                4. Click **Predict Data**.
                5. Results will appear in the table below.
                6. Download the prediction results as Excel.
            """)
            with open('data_template.xlsx', 'rb') as file:
                st.download_button(
                    label='Download Template',
                    data=file,
                    file_name='data_template.xlsx',
                    mime='application/vnd.ms-excel'
                )

        uploaded_file = st.file_uploader('Upload Student Data', type=['xlsx', 'xls'])

        if uploaded_file is not None:
            up = pd.read_excel(uploaded_file)
            up['ID'] = up['ID'].astype(str)

            st.write('')
            preview = st.slider('**Preview Rows**', 1, len(up), 2)
            st.dataframe(up.head(preview))

            df_up = pd.DataFrame(up, columns=[
                'ID', 'Name', 'Marital Status', 'Application Mode',
                'Previous Qualification Grade', 'Admission Grade',
                'Displaced', 'Debtor', 'Tuition up to date', 'Gender', 'Scholarship',
                'Age at Enrollment', 'Units 1st Semester Enrolled',
                'Units 1st Semester Approved', 'Units 1st Semester Grade',
                'Units 2nd Semester Enrolled', 'Units 2nd Semester Approved',
                'Units 2nd Semester Grade', 'Units 2nd Semester Evaluations',
                'Units 2nd Semester No Evaluations'
            ])

            df_up.rename(columns={
                'Marital Status': 'Marital_status',
                'Application Mode': 'Application_mode',
                'Previous Qualification Grade': 'Previous_qualification_grade',
                'Admission Grade': 'Admission_grade',
                'Tuition up to date': 'Tuition_fees_up_to_date',
                'Scholarship': 'Scholarship_holder',
                'Age at Enrollment': 'Age_at_enrollment',
                'Units 1st Semester Enrolled': 'Curricular_units_1st_sem_enrolled',
                'Units 1st Semester Approved': 'Curricular_units_1st_sem_approved',
                'Units 1st Semester Grade': 'Curricular_units_1st_sem_grade',
                'Units 2nd Semester Enrolled': 'Curricular_units_2nd_sem_enrolled',
                'Units 2nd Semester Approved': 'Curricular_units_2nd_sem_approved',
                'Units 2nd Semester Grade': 'Curricular_units_2nd_sem_grade',
                'Units 2nd Semester Evaluations': 'Curricular_units_2nd_sem_evaluations',
                'Units 2nd Semester No Evaluations': 'Curricular_units_2nd_sem_without_evaluations'
            }, inplace=True)

            student_ids = df_up.pop('ID')
            student_names = df_up.pop('Name')

            df_up['Gender'] = df_up['Gender'].map(GENDER_MAP)
            df_up['Marital_status'] = df_up['Marital_status'].map(MARITAL_MAP)
            df_up['Application_mode'] = df_up['Application_mode'].map(APPLICATION_MAP)

            if st.button('Predict Data'):
                df_input = data_preprocessing(df_up, False, len(up))
                output = model_predict(df_input)

                result = pd.DataFrame({
                    'ID': student_ids,
                    'Name': student_names,
                    'Status': ['Graduate' if p == 1 else 'Dropout' for p in output]
                })

                st.write('')
                st.write('**Results**')
                st.dataframe(result.style.applymap(color_mapping, subset=['Status']))

                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    result.to_excel(writer, sheet_name='Prediction', index=False)

                st.download_button(
                    label='Download Prediction',
                    data=buffer.getvalue(),
                    file_name='Student Data Prediction.xlsx',
                    mime='application/vnd.ms-excel'
                )

    st.write('')

if __name__ == '__main__':
    main()