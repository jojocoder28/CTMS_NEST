data=pd.read_excel("../test_data.xlsx")
data_copy=data.copy()
data.head()
date_columns = ['Start Date', 'Primary Completion Date', 'Completion Date', 'Last Update Posted']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])

data['Start_Year'] = data['Start Date'].dt.year
data['Start_Month'] = data['Start Date'].dt.month
data['Study_Duration'] = (data['Completion Date'] - data['Start Date']).dt.days
data['Late_Study'] = (data['Completion Date'] - data['Primary Completion Date']).dt.days
data['Days_Since_Started'] = (pd.Timestamp.now() - data['Start Date']).dt.days
data['Start_Season'] = data['Start Date'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring

df=data.copy()
study_design_split = df['Study Design'].str.split('|', expand=True)
df['Allocation'] = study_design_split[0]
df['Intervention Model'] = study_design_split[1]
df['Primary Purpose'] = study_design_split[2]
df['Masking'] = study_design_split[3]

df = df.drop(columns=['Study Design'])
cleaned_data=df.drop(columns=['Other Outcome Measures', 'Collaborators', 'Results First Posted','NCT Number','Primary Outcome Measures','Secondary Outcome Measures','Sponsor','Study Title','Study URL','Brief Summary','Other IDs','Start Date','Primary Completion Date','Completion Date','First Posted','Last Update Posted','Start_Year', 'Start_Month','Days_Since_Started', 'Start_Season', 'Late_Study'])
label_encoders = load('./Saved Encoders/label_encoders_method2.joblib')

cleaned_data_categorical_columns = cleaned_data.select_dtypes(include=['object']).columns

for col in cleaned_data_categorical_columns:
    if col in label_encoders:
        le = label_encoders[col]
        le_classes = set(le.classes_)
        new_classes = set(cleaned_data[col].unique()) - le_classes
        
        if new_classes:
            le.classes_ = np.append(le.classes_, list(new_classes))
        
        cleaned_data[col] = le.transform(cleaned_data[col])
    else:
        print(f"Warning: Column '{col}' was not present in the original data and cannot be encoded.")

model=load("./Saved Models/lightgbm_model_method2.joblib")
y_pred=model.predict(cleaned_data.values)
print(y_pred)

data_copy["Recruitment Rate (RR)"]=[round(num, 2) for num in y_pred]





######## Previous Working Code ###########


import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import plotly.express as px

label_encoders = load('./label_encoders_method2.joblib')
model = load('./lightgbm_model_method2.joblib')
data = pd.read_excel("./test_data.xlsx")
saved_data=data.copy()
data['Current Recruitment Rate(CRR)'] = 0.0
flag=10
def preprocess(data):
    date_columns = ['Start Date', 'Primary Completion Date', 'Completion Date', 'Last Update Posted']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])

    data['Start_Year'] = data['Start Date'].dt.year
    data['Start_Month'] = data['Start Date'].dt.month
    data['Study_Duration'] = (data['Completion Date'] - data['Start Date']).dt.days
    data['Late_Study'] = (data['Completion Date'] - data['Primary Completion Date']).dt.days
    data['Days_Since_Started'] = (pd.Timestamp.now() - data['Start Date']).dt.days
    data['Start_Season'] = data['Start Date'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring

    df = data.copy()

    study_design_split = df['Study Design'].str.split('|', expand=True)
    df['Allocation'] = study_design_split[0]
    df['Intervention Model'] = study_design_split[1]
    df['Primary Purpose'] = study_design_split[3]
    df['Masking'] = study_design_split[2]

    df = df.drop(columns=['Study Design'])

    data = df.copy()
    return data

def load_data(data,flag):
    data=data.head(flag)
    data_copy = data.copy()
    
    
    # try:
    cleaned_data = data_copy.drop(columns=['Other Outcome Measures', 'Collaborators', 'Results First Posted','NCT Number',
                                    'Primary Outcome Measures','Secondary Outcome Measures','Sponsor','Study Title',
                                    'Study URL','Brief Summary','Other IDs','Start Date','Primary Completion Date',
                                    'Completion Date','First Posted','Last Update Posted','Start_Year', 'Start_Month',
                                    'Days_Since_Started', 'Start_Season', 'Late_Study',"Current Recruitment Rate(CRR)"],errors='ignore')
    
    
    cleaned_data_categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
    for col in cleaned_data_categorical_columns:
        if col in label_encoders:
            le = label_encoders[col]
            le_classes = set(le.classes_)
            new_classes = set(cleaned_data[col].unique()) - le_classes
            
            if new_classes:
                le.classes_ = np.append(le.classes_, list(new_classes))
            
            cleaned_data[col] = le.transform(cleaned_data[col])
        else:
            st.warning(f"Column '{col}' was not present in the original data and cannot be encoded.")
    
    return cleaned_data, data_copy


def create_dashboard(data):
    # Title of the app
    st.title("Clinical Study Dashboard")

    # Display basic information about the dataset
    st.write("## Overview of Clinical Studies")
    st.write("The dataset contains clinical study information including NCT number, study title, conditions, outcomes, etc.")

    # Gender filter
    gender_filter = st.multiselect('Select Gender(s)', options=data['Sex'].unique(), default=data['Sex'].unique())
    filtered_data = data[data['Sex'].isin(gender_filter)]

    # Study Status filter
    status_filter = st.multiselect('Select Study Status', options=filtered_data['Study Status'].unique(), default=filtered_data['Study Status'].unique())
    filtered_data = filtered_data[filtered_data['Study Status'].isin(status_filter)]

    # Study Results filter
    results_filter = st.multiselect('Select Study Results', options=filtered_data['Study Results'].unique(), default=filtered_data['Study Results'].unique())
    filtered_data = filtered_data[filtered_data['Study Results'].isin(results_filter)]

    # Age filter with bins
    # age_bins = st.slider('Select Age Range', min_value=int(filtered_data['Age'].min()), max_value=int(filtered_data['Age'].max()), value=(int(filtered_data['Age'].min()), int(filtered_data['Age'].max())))
    # filtered_data = filtered_data[(filtered_data['Age'] >= age_bins[0]) & (filtered_data['Age'] <= age_bins[1])]

    # Display filtered dataset
    st.write("### Filtered Data", filtered_data)

    # Create a bar plot for study status distribution
    status_count = filtered_data['Study Status'].value_counts()
    fig1 = px.bar(
        x=status_count.index,
        y=status_count.values,
        labels={'x': 'Study Status', 'y': 'Count'},
        title="Study Status Distribution"
    )
    st.plotly_chart(fig1)

    # Create a pie chart for study phases distribution
    phase_count = filtered_data['Phases'].value_counts()
    fig2 = px.pie(
        names=phase_count.index,
        values=phase_count.values,
        title="Study Phases Distribution"
    )
    st.plotly_chart(fig2)

    # Convert 'Start Date' to Period (monthly)
    filtered_data['Start Date'] = pd.to_datetime(filtered_data['Start Date'], errors='coerce')
    filtered_data['Start Date Period'] = filtered_data['Start Date'].dt.to_period('M')

    # Convert Period back to Timestamp (datetime) for Plotly compatibility
    filtered_data['Start Date Timestamp'] = filtered_data['Start Date Period'].dt.start_time

    # Group by the newly created 'Start Date Timestamp' column
    enrollment_data = filtered_data.groupby('Start Date Timestamp').size().reset_index(name='Enrollment Count')

    # Create a line plot for study enrollment over time
    fig3 = px.line(
        enrollment_data, 
        x='Start Date Timestamp', 
        y='Enrollment Count',
        title="Study Enrollment Over Time"
    )
    st.plotly_chart(fig3)


    # Filter out records where 'Start Date' or 'Completion Date' are missing
    filtered_data = filtered_data.dropna(subset=['Start Date', 'Completion Date'])

    # Create a scatter plot of enrollment against study duration
    filtered_data['Completion Date'] = pd.to_datetime(filtered_data['Completion Date'], errors='coerce')
    filtered_data['Study Duration'] = (filtered_data['Completion Date'] - filtered_data['Start Date']).dt.days
    fig4 = px.scatter(
        filtered_data, 
        x='Enrollment', 
        y='Study Duration', 
        color='Study Status',
        labels={'Enrollment': 'Enrollment', 'Study Duration': 'Study Duration (Days)'},
        title="Enrollment vs. Study Duration"
    )
    st.plotly_chart(fig4)

data = preprocess(data)
cleaned_data, data_copy = load_data(data,flag)

# Predict Recruitment Rate (RR)
y_pred = model.predict(cleaned_data.values)
data_copy["Recruitment Rate (RR)"] = [round(num, 2) for num in y_pred]

# Title of the app

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["CTMS", "Manage Trials", "Data Dashboard"])

# Dashboard Page
if page == "CTMS":
    st.title("Clinical Trial Management System (CTMS)")

    if "data_copy" not in st.session_state:
        st.session_state.data_copy = data_copy.copy()
    data_copy=st.session_state.data_copy.copy()
    trial_list = st.session_state.data_copy["Study Title"].tolist()
    selected_trial = st.sidebar.selectbox("Select a Trial", trial_list)
    
    if selected_trial:
        trial_data = data_copy[data_copy["Study Title"] == selected_trial].iloc[0]
        
        st.subheader(f"Trial: {trial_data['Study Title']}")
        st.write(f"**Study Status:** {trial_data['Study Status']}")
        st.write(f"**Study Results:** {trial_data['Study Results']}")
        st.write(f"**Conditions:** {trial_data['Conditions']}")
        st.write(f"**Interventions:** {trial_data['Interventions']}")
        st.write(f"**Enrollment:** {trial_data['Enrollment']}")
        st.write(f"**Study Duration:** {trial_data['Study_Duration']} days")
        st.write(f"**Recruitment Rate (RR):** {trial_data['Recruitment Rate (RR)']}")
        
        # Risk alerts
        if trial_data['Study Status'] in ["RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"] and trial_data['Study Results'] == "NO": 
          st.subheader("Recruitment Risk Alert")
          recruitment_rate = trial_data["Recruitment Rate (RR)"]
          if recruitment_rate > trial_data['Current Recruitment Rate(CRR)']:
              st.error(f"⚠️ Recruitment rate is {trial_data['Current Recruitment Rate(CRR)']} (At Risk)")
          else:
              st.success(f"✅ Recruitment rate is {trial_data['Current Recruitment Rate(CRR)']} (Healthy)")
        
        # Edit Button
        if st.button("Edit Trial Data"):
            trial_index = data_copy[data_copy["Study Title"] == selected_trial].index[0]
            temp_df=data_copy[data_copy["Study Title"] == selected_trial]
            st.session_state.selected_trial_index = trial_index
            st.session_state.temp_df=temp_df
            st.session_state.selected_trial = selected_trial
            st.query_params.trial=trial_index
            st.rerun()
            
            
# Manage Trials Page
elif page == "Manage Trials":
    st.header("Manage Clinical Trials")
    
    if "data_copy" not in st.session_state:
        st.session_state.data_copy = data_copy.copy()
        
    # Form to add new trial data
    with st.form(key='new_trial_form'):
        study_status_options = data['Study Status'].dropna().unique()
        study_results_options = data['Study Results'].dropna().unique()
        sex_options = data['Sex'].dropna().unique()
        funder_type_options = data['Funder Type'].dropna().unique()
        study_type_options = data['Study Type'].dropna().unique()
        allocation_options = data['Allocation'].dropna().unique()
        intervention_model_options = data['Intervention Model'].dropna().unique()
        primary_purpose_options = data['Primary Purpose'].dropna().unique()
        masking_options = data['Masking'].dropna().unique()
        conditions_data = data['Conditions'].str.split('|').explode().unique()
        interventions_data = data['Interventions'].str.split('|').explode().unique()
        today = datetime.today().date()

        # User input fields
        NCT_Number = st.text_input("NCT Number")
        # Current_Recruitment_Rate = sr.text_input("Current Recruitment Rate(CRR)")
        trial_name = st.text_input("Study Title")
        start_date = st.date_input("Start Date")
        primary_completion_date = st.date_input("Primary Completion Date")
        completion_date = st.date_input("Completion Date")
        last_update_posted = st.date_input("Last Update Posted")
        
      
        
        study_status = st.selectbox('Study Status', study_status_options)
        study_results = st.selectbox('Study Results', study_results_options)
        conditions = st.multiselect('Conditions', conditions_data)
        interventions = st.multiselect('Interventions', interventions_data)
        sex = st.selectbox('Sex', sex_options)
        age = st.number_input('Age', min_value=18, max_value=100, step=1)
        phases = st.selectbox('Phases', data['Phases'].dropna().unique())
        enrollment = st.number_input('Enrollment', min_value=1, max_value=10000, step=1)
        funder_type = st.selectbox('Funder Type', funder_type_options)
        study_type = st.selectbox('Study Type', study_type_options)
        locations = st.text_input('Locations')
        study_duration = st.number_input('Study Duration (days)', min_value=1, max_value=10000, step=1)
        allocation = st.selectbox('Allocation', allocation_options)
        intervention_model = st.selectbox('Intervention Model', intervention_model_options)
        primary_purpose = st.selectbox('Primary Purpose', primary_purpose_options)
        masking = st.selectbox('Masking', masking_options)
            
        # Submit button
        submit_button = st.form_submit_button("Submit Trial")
        
        if submit_button:
            Current_recruitment_rate = enrollment / (today - start_date).days * 30
            # Prepare the new trial data
            new_trial = {
                "NCT Number": NCT_Number,
                "Study Title": trial_name,
                "Start Date": start_date,
                "Primary Completion Date": primary_completion_date,
                "Completion Date": completion_date,
                "Last Update Posted": last_update_posted,
                "Current Recruitment Rate(CRR)": Current_recruitment_rate,
                'Study Status': study_status,
                'Study Results': study_results,
                'Conditions': '|'.join(conditions),
                'Interventions': '|'.join(interventions),
                'Sex': sex,
                'Age': age,
                'Phases': phases,
                'Enrollment': enrollment,
                'Funder Type': funder_type,
                'Study Type': study_type,
                'Locations': locations,
                'Study_Duration': study_duration,
                # 'Allocation': allocation,
                # 'Intervention Model': intervention_model,
                # 'Primary Purpose': primary_purpose,
                # 'Masking': masking
                'Study Design': allocation+'|'+intervention_model+'|'+masking+'|'+primary_purpose
            }
            
            # Append the new trial to the data (Simulate adding to a database or data file)
            new_trial_df = pd.DataFrame([new_trial])
            new_trial_df2 = preprocess(new_trial_df)
            # Reprocess the new trial for prediction
            # cleaned_data_new_trial = cleaned_data.append(new_trial, ignore_index=True)
            cleaned_data_new_trial = pd.concat([cleaned_data,new_trial_df], ignore_index=True)
            cleaned_data_new_trial, _ = load_data(preprocess(cleaned_data_new_trial),flag)
            flag+=1
            y_pred_new = model.predict(cleaned_data_new_trial.values)
            new_trial_df2["Recruitment Rate (RR)"] = round(y_pred_new[-1], 2)
            date_columns = ['Start Date', 'Primary Completion Date', 'Completion Date']
            for col in date_columns:
                if col in data_copy.columns:
                    data_copy[col] = data_copy[col].astype(str)

            # data_copy = pd.concat([data_copy, new_trial_df2], ignore_index=True)
            st.session_state.data_copy = pd.concat([st.session_state.data_copy, new_trial_df2], ignore_index=True)
            st.success(f"New trial '{trial_name}' added successfully!")
            # st.dataframe(st.session_state.data_copy)  # Show updated data with new trial

elif page == "Data Dashboard":
    create_dashboard(saved_data)


trial_param = st.query_params.get("trial")
if trial_param:
    trial_index = int(trial_param)
    trial_data = st.session_state.data_copy.iloc[trial_index]
    
        
    print(trial_data)

    st.header(f"Trial Details: {trial_data['Study Title']}")

    new_status = st.selectbox("Study Status", data["Study Status"].dropna().unique(), 
                              index=list(data["Study Status"].dropna().unique()).index(trial_data["Study Status"]))
    # new_rr = st.number_input("Recruitment Rate (RR)", min_value=0.0, max_value=100.0, 
    #                          value=trial_data["Recruitment Rate (RR)"], step=0.1)
    new_enrollment = st.number_input("Enrollment", min_value=0, value=int(trial_data.get("Enrollment", 0)), step=1)
    new_study_result = st.selectbox("Study Results", ['YES','NO'])
    new_completion_time = st.number_input("Completion Time (days)", min_value=0, 
                                          value=int(trial_data.get("Study_Duration", 0)), step=1)

    if st.button("Save Changes"):
        st.session_state.data_copy.at[trial_index, "Study Status"] = new_status
        # st.session_state.data_copy.at[trial_index, "Recruitment Rate (RR)"] = new_rr
        st.session_state.data_copy.at[trial_index, "Enrollment"] = new_enrollment
        st.session_state.data_copy.at[trial_index, "Study Results"] = new_study_result
        st.session_state.data_copy.at[trial_index, "Study_Duration"] = new_completion_time
        st.success("Changes saved successfully!")

    if st.button("Back to Dashboard"):
        del st.session_state.selected_trial
        st.rerun()