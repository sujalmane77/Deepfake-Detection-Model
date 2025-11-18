import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

attendance_df = pd.read_csv("attendance_logs.csv")
events_df = pd.read_csv("event_participation.csv")
lms_df = pd.read_csv("lms_usage.csv")

st.title("📊 Smart Campus Insights")
st.sidebar.header("🔍 Filters")

students = attendance_df['StudentID'].unique()
selected_students = st.sidebar.multiselect("Select Students", students, default=students)

filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

st.subheader("📋 Attendance Trends")
attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
st.line_chart(attendance_summary)

st.subheader("🎓 Event Participation")
event_counts = filtered_events['EventName'].value_counts()
st.bar_chart(event_counts)

st.subheader("💻 LMS Usage Patterns")
lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
st.dataframe(lms_summary)

st.subheader("🤖 Predict Student Engagement Risk")

ml_data = pd.merge(attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
                   lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
                   on='StudentID')

ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y = ml_data['Engagement']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Model Performance:")
st.text(classification_report(y_test, y_pred))

st.subheader("📈 Predict Engagement for New Student")
absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

if st.button("Predict Engagement"):
    prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
    result = "Engaged" if prediction[0] == 1 else "At Risk"
    st.success(f"Predicted Engagement Status: {result}")
