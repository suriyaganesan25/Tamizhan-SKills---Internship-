import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Student Performance Analytics Dashboard")

uploaded_file = st.file_uploader("Upload your student performance CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    st.subheader("📄 Student Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    st.subheader("📊 Average Performance")
    avg_marks = df["Marks"].mean()
    avg_attendance = df["Attendance"].mean()
    avg_logins = df["Logins"].mean()

    st.metric("Average Marks", f"{avg_marks:.2f}")
    st.metric("Average Attendance", f"{avg_attendance:.2f}%")
    st.metric("Average Logins", f"{avg_logins:.0f}")

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[["Marks", "Attendance", "Logins"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🏆 Top 5 Students by Marks")
    st.dataframe(df.nlargest(5, "Marks"))

    st.subheader("⚠️ Bottom 5 Students by Marks (At Risk)")
    st.dataframe(df.nsmallest(5, "Marks"))

    st.subheader("📊 Marks Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Marks"], kde=True, bins=10, ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Attendance vs Marks")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Attendance", y="Marks", hue="Marks", palette="coolwarm", ax=ax)
    st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV file to view analytics.")
