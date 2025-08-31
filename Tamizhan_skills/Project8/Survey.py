import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Sample advanced survey data
data = {
    'Age Group': ['18-25', '26-35', '36-45', '18-25', '26-35', '36-45', '46-55', '46-55', '18-25', '26-35', '36-45', '46-55'],
    'Favorite Product': ['A', 'B', 'A', 'C', 'B', 'B', 'C', 'A', 'C', 'B', 'A', 'C'],
    'Satisfaction Rating': [4, 5, 3, 4, 4, 2, 5, 3, 4, 5, 3, 4],
    'Recommend Likelihood': [5, 4, 3, 4, 5, 2, 4, 3, 5, 4, 3, 5],
    'Usage Frequency': ['Daily', 'Weekly', 'Monthly', 'Weekly', 'Daily', 'Monthly', 'Weekly', 'Daily', 'Monthly', 'Daily', 'Weekly', 'Monthly']
}
df = pd.DataFrame(data)

# Set styles
sns.set(style="whitegrid")

# Function: Grouped bar chart of categorical responses by demographic
def plot_grouped_bar(data, x_col, hue_col, title, pdf=None):
    plt.figure(figsize=(10,6))
    ax = sns.countplot(data=data, x=x_col, hue=hue_col, palette='Set2')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel('Count')
    plt.legend(title=hue_col)
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.show()

# Function: Pie chart for distribution of single categorical variable
def plot_pie(data, column, title, pdf=None):
    counts = data[column].value_counts()
    plt.figure(figsize=(7,7))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title(title)
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.show()

# Function: Distribution plots for rating scales
def plot_rating_distributions(data, rating_columns, pdf=None):
    plt.figure(figsize=(12,5))
    for i, col in enumerate(rating_columns):
        plt.subplot(1, len(rating_columns), i+1)
        sns.histplot(data[col], bins=5, kde=False, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Rating')
        plt.ylabel('Count')
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.show()

# Function: Correlation heatmap of numeric columns
def plot_correlation_heatmap(data, num_cols, pdf=None):
    corr = data[num_cols].corr()
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    if pdf:
        pdf.savefig()
    plt.show()

# Save multiple plots to a PDF
with PdfPages('Survey_Analysis_Report.pdf') as pdf:
    plot_grouped_bar(df, 'Favorite Product', 'Age Group', 'Favorite Products by Age Group', pdf)
    plot_pie(df, 'Usage Frequency', 'Usage Frequency Distribution', pdf)
    plot_rating_distributions(df, ['Satisfaction Rating', 'Recommend Likelihood'], pdf)
    plot_correlation_heatmap(df, ['Satisfaction Rating', 'Recommend Likelihood'], pdf)

print("PDF report 'Survey_Analysis_Report.pdf' generated with all plots.")

# Advanced: Interactive Plotly dashboard example
fig = make_subplots(rows=2, cols=2, 
                    specs=[[{"type":"bar"}, {"type":"pie"}],
                           [{"type":"histogram"}, {"type":"heatmap"}]],
                    subplot_titles=("Favorite Products by Age Group", "Usage Frequency Distribution",
                                    "Satisfaction Rating Distribution", "Correlation Heatmap"))

# Bar chart
bar_data = df.groupby(['Age Group', 'Favorite Product']).size().reset_index(name='count')
for age_group in df['Age Group'].unique():
    subset = bar_data[bar_data['Age Group'] == age_group]
    fig.add_trace(go.Bar(x=subset['Favorite Product'], y=subset['count'], name=str(age_group)), row=1, col=1)

# Pie chart
usage_counts = df['Usage Frequency'].value_counts()
fig.add_trace(go.Pie(labels=usage_counts.index, values=usage_counts.values, name="Usage Frequency"), row=1, col=2)

# Histogram
fig.add_trace(go.Histogram(x=df['Satisfaction Rating'], nbinsx=5, name='Satisfaction Rating'), row=2, col=1)

# Correlation heatmap
corr = df[['Satisfaction Rating', 'Recommend Likelihood']].corr().values
fig.add_trace(go.Heatmap(z=corr, x=['Satisfaction Rating', 'Recommend Likelihood'], y=['Satisfaction Rating', 'Recommend Likelihood'],
                         colorscale='RdBu', zmin=-1, zmax=1, showscale=True), row=2, col=2)

fig.update_layout(title_text="Advanced Survey Data Dashboard", height=800, width=900)
fig.show()
