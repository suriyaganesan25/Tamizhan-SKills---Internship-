import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Netflix Userbase.csv')

# Convert date columns to datetime
df['Join Date'] = pd.to_datetime(df['Join Date'])
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])

# Analyze Monthly Revenue distribution
plt.figure(figsize=(10,5))
sns.histplot(df['Monthly Revenue'], bins=30, kde=True)
plt.title('Monthly Revenue Distribution')
plt.xlabel('Monthly Revenue ($)')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()

# Analyze Subscription Types
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Subscription Type')
plt.title('Subscription Type Counts')
plt.xlabel('Subscription Type')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()

# Analyze user joining trends over time
df.set_index('Join Date', inplace=True)
monthly_joins = df.resample('M').size()

plt.figure(figsize=(12,6))
monthly_joins.plot()
plt.title('User Joins Over Time (Monthly)')
plt.xlabel('Join Date')
plt.ylabel('Number of New Users')
plt.tight_layout()
plt.show()
df.reset_index(inplace=True)

# Age distribution
plt.figure(figsize=(10,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()

# Gender breakdown
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.tight_layout()
plt.show()

# Device usage counts
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Device', order=df['Device'].value_counts().index)
plt.title('Device Usage for Netflix')
plt.xlabel('Device')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plan Duration distribution
plt.figure(figsize=(10,5))
sns.histplot(df['Plan Duration'], bins=20)
plt.title('Plan Duration Distribution (months)')
plt.xlabel('Plan Duration')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()
