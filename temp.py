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