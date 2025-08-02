# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

def main():
    st.title("ABC Manufacturing – Retail Sales Analysis & Forecasting")

    # Upload file
    uploaded_file = st.file_uploader("Upload your Retail_sales.csv file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📌 First 5 Rows of the Dataset")
        st.dataframe(df.head())

        st.subheader("📌 Dataset Overview")
        st.text("Structure:")
        st.text(df.info())
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Descriptive Statistics:")
        st.dataframe(df.describe())

        # Preprocessing
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month_name()

        # --- Column Chart ---
        st.subheader("📊 Total Sales Revenue by Product Category")
        fig1, ax1 = plt.subplots()
        product_revenue = df.groupby('Product Category')['Sales Revenue (USD)'].sum().sort_values(ascending=False)
        sns.barplot(x=product_revenue.index, y=product_revenue.values, palette='viridis', ax=ax1)
        ax1.set_title('Total Sales Revenue by Product Category')
        ax1.set_xlabel('Product Category')
        ax1.set_ylabel('Sales Revenue (USD)')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        # --- Line Chart ---
        st.subheader("📈 Monthly Sales Revenue Fluctuation")
        fig2, ax2 = plt.subplots()
        monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales Revenue (USD)'].sum().reset_index()
        monthly_sales['Date'] = monthly_sales['Date'].astype(str)
        sns.lineplot(x='Date', y='Sales Revenue (USD)', data=monthly_sales, marker='o', ax=ax2, color='skyblue')
        ax2.set_title('Monthly Sales Revenue')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Sales Revenue (USD)')
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        # --- Pie Chart ---
        st.subheader("🧭 Sales Distribution by Store Location")
        fig3, ax3 = plt.subplots()
        location_sales = df.groupby('Store Location')['Sales Revenue (USD)'].sum().sort_values(ascending=False)
        top_n = 10
        if len(location_sales) > top_n:
            other_sales = location_sales.iloc[top_n:].sum()
            location_sales = location_sales.iloc[:top_n]
            location_sales['Other'] = other_sales
        ax3.pie(location_sales, labels=location_sales.index, autopct='%1.1f%%', startangle=140)
        ax3.set_title('Percentage Share of Sales by Store Location')
        st.pyplot(fig3)

        # --- Heatmap ---
        st.subheader("🧪 Correlation Matrix of Numerical Variables")
        fig4, ax4 = plt.subplots()
        num_df = df.select_dtypes(include=['int64', 'float64', 'bool'])
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        st.pyplot(fig4)

        # --- Box Plot ---
        st.subheader("📦 Sales Revenue Distribution by Product Category")
        fig5, ax5 = plt.subplots()
        sns.boxplot(x='Product Category', y='Sales Revenue (USD)', data=df, palette='Set3', ax=ax5)
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
        st.pyplot(fig5)

        # --- Linear Regression Model ---
        st.subheader("📉 Linear Regression – Sales Prediction")
        df_model = df.drop(['Store ID', 'Product ID', 'Date', 'Month'], axis=1)
        X = df_model.drop('Sales Revenue (USD)', axis=1)
        y = df_model['Sales Revenue (USD)']
        cat_cols = X.select_dtypes(include=['object', 'bool']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X_processed = pd.concat([X[num_cols].reset_index(drop=True), encoded_df], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.metric(label="Mean Squared Error", value=f"{mse:.2f}")
        st.metric(label="Root Mean Squared Error", value=f"{rmse:.2f}")
        st.metric(label="R² Score", value=f"{r2:.2f}")

        # Show top coefficients
        coef_df = pd.DataFrame({
            'Feature': X_processed.columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values(by='Abs_Coefficient', ascending=False)
        st.write("Top Influential Features:")
        st.dataframe(coef_df.head(10))


if __name__ == '__main__':
    main()
