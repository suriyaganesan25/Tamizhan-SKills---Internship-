import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("owid-covid-data.csv")

df = df[["location", "date", "total_cases", "total_deaths", "total_tests", "population"]]

df["date"] = pd.to_datetime(df["date"])

df = df.dropna(subset=["total_cases"])

countries = ["United States", "India", "Brazil", "Russia", "United Kingdom"]
df = df[df["location"].isin(countries)]

plt.figure(figsize=(12, 6))
for country in countries:
    country_data = df[df["location"] == country]
    plt.plot(country_data["date"], country_data["total_cases"], label=country)

plt.title("COVID-19 Total Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.legend()
plt.grid(True)
plt.show()

df["death_rate"] = (df["total_deaths"] / df["total_cases"]) * 100

latest_data = df.loc[df.groupby("location")["date"].idxmax()]

heatmap_data = latest_data.pivot_table(index="location", values="death_rate")
plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".2f")
plt.title("COVID-19 Death Rate (%) - Latest Data")
plt.show()

latest_data["tests_per_million"] = (latest_data["total_tests"] / latest_data["population"]) * 1e6
latest_data_sorted = latest_data.sort_values(by="tests_per_million", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=latest_data_sorted, x="tests_per_million", y="location", palette="Blues_d")
plt.title("COVID-19 Tests per Million Population")
plt.xlabel("Tests per Million")
plt.ylabel("Country")
plt.show()
