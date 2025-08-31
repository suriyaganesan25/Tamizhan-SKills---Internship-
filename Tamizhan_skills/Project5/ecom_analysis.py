import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("synthetic_ecommerce_data.csv")

df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

top_products = (
    df.groupby('Product_ID')['Revenue']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_products, x='Revenue', y='Product_ID', palette='viridis', hue=None, legend=False)
plt.title('Top 10 Products by Revenue')
plt.xlabel('Revenue')
plt.ylabel('Product ID')
plt.tight_layout()
plt.show()
--
df['Hour'] = df['Transaction_Date'].dt.hour
sales_by_hour = df.groupby('Hour')['Revenue'].sum()

plt.figure(figsize=(10, 6))
sns.lineplot(x=sales_by_hour.index, y=sales_by_hour.values, marker='o')
plt.title('Revenue by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Revenue')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

user_activity = df.groupby('Customer_ID')['Transaction_Date'].nunique()
retained_users = (user_activity > 1).sum()
new_users = (user_activity == 1).sum()

plt.figure(figsize=(6, 6))
plt.pie([retained_users, new_users],
        labels=['Returning Users', 'New Users'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#4daf4a', '#377eb8'])
plt.title('User Retention')
plt.tight_layout()
plt.show()

category_sales = (
    df.groupby('Category')['Revenue']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(data=category_sales, x='Revenue', y='Category', palette='cubehelix', hue=None, legend=False)
plt.title('Sales by Category')
plt.xlabel('Revenue')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

region_sales = (
    df.groupby('Region')['Revenue']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(data=region_sales, x='Revenue', y='Region', palette='coolwarm', hue=None, legend=False)
plt.title('Sales by Region')
plt.xlabel('Revenue')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

print("✅ Analysis complete! Dashboard visuals generated.")
