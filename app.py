# app.py

from google.colab import files
uploaded = files.upload()

# Step 1: Data Collection – ABC Manufacturing (Retail Sales Data Integration)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the Retail_sales.csv file into a DataFrame
df = pd.read_csv('Retail_sales.csv')

# Display first 5 rows of the dataset
print("📌 First 5 rows of the dataset:")
print(df.head())

# Basic structure of the dataset
print("\n📌 Dataset Information:")
df.info()

# Check for missing values
print("\n📌 Missing Values:")
print(df.isnull().sum())

# Basic statistical summary
print("\n📌 Descriptive Statistics:")
print(df.describe())

# Show the number of unique values per column
print("\n📌 Unique Values in Each Column:")
print(df.nunique())

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Extract month for Line Chart
df['Month'] = df['Date'].dt.month_name()

# --- Column Chart ---
plt.figure(figsize=(12, 7))
product_revenue = df.groupby('Product Category')['Sales Revenue (USD)'].sum().sort_values(ascending=False)
sns.barplot(x=product_revenue.index, y=product_revenue.values, palette='viridis')
plt.title('Total Sales Revenue by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Revenue (USD)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Line Chart ---
plt.figure(figsize=(14, 7))
monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales Revenue (USD)'].sum().reset_index()
monthly_sales['Date'] = monthly_sales['Date'].astype(str)
sns.lineplot(x='Date', y='Sales Revenue (USD)', data=monthly_sales, marker='o', color='skyblue')
plt.title('Monthly Sales Revenue Fluctuation')
plt.xlabel('Month')
plt.ylabel('Total Sales Revenue (USD)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Pie Chart ---
plt.figure(figsize=(10, 10))
location_sales = df.groupby('Store Location')['Sales Revenue (USD)'].sum().sort_values(ascending=False)
top_n = 10
if len(location_sales) > top_n:
    other_sales = location_sales.iloc[top_n:].sum()
    location_sales = location_sales.iloc[:top_n]
    location_sales['Other'] = other_sales
cmap = plt.cm.get_cmap('tab20')
colors = [cmap(i / len(location_sales)) for i in range(len(location_sales))]
wedges, texts, autotexts = plt.pie(location_sales, labels=location_sales.index, autopct='%1.1f%%',
                                   startangle=140, colors=colors, pctdistance=0.75,
                                   labeldistance=1.1, textprops={'fontsize': 10, 'color': 'black'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)
plt.title('Percentage Share of Sales by Store Location')
plt.axis('equal')
plt.tight_layout()
plt.show()

# --- Heatmap ---
plt.figure(figsize=(10, 8))
numerical_df = df.select_dtypes(include=['int64', 'float64', 'bool'])
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Box Plot ---
plt.figure(figsize=(14, 8))
sns.boxplot(x='Product Category', y='Sales Revenue (USD)', data=df, palette='Set3')
plt.title('Distribution of Sales Revenue by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Sales Revenue (USD)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Linear Regression Model ---
df_model = df.drop(['Store ID', 'Product ID', 'Date', 'Month'], axis=1)
X = df_model.drop('Sales Revenue (USD)', axis=1)
y = df_model['Sales Revenue (USD)']
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
X_processed = pd.concat([X[numerical_cols], encoded_df], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
coefficients = pd.DataFrame({'Feature': X_processed.columns, 'Coefficient': model.coef_})
coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
print(coefficients.sort_values(by='Abs_Coefficient', ascending=False).head(10))