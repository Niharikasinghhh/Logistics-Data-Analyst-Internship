# Logistics Data Analysis - Week 1

## Project Overview

This project is part of a Virtual Logistics Data Analyst Internship.

The objective of Week 1 is to develop a strategic plan for analyzing logistics delivery performance and identifying factors that contribute to delivery delays.

## Objectives

- Analyze logistics delivery performance
- Identify important logistics KPIs
- Explore factors affecting delivery delays
- Propose predictive analytics techniques
- Develop a strategic roadmap for logistics analysis

## KPIs

- On-Time Delivery Rate
- Delivery Delay Rate
- Average Delivery Time
- Transportation Cost per Shipment

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib

## Proposed Analysis Workflow

Data Collection → Data Cleaning → Exploratory Data Analysis → KPI Analysis → Predictive Modeling → Business Insights

## Repository Contents

- `logistics_analysis.py` - Python code illustrating the proposed analytical approach
- Week 1 strategic planning report




# Week 2: Data Collection, Cleaning and Preprocessing

## Project Overview

This week focuses on preparing logistics data for further analysis. The objective is to simulate a complete data preprocessing pipeline and improve data quality before performing exploratory data analysis and predictive modeling.

## Objectives

- Load and inspect logistics data
- Identify and handle missing values
- Detect and remove duplicate records
- Convert date columns into datetime format
- Create logistics-related features
- Detect potential outliers using the IQR method
- Normalize numerical data for future machine learning models
- Save the cleaned dataset for further analysis

## Data Preprocessing Steps

### 1. Data Collection

A publicly available logistics and e-commerce dataset is used as a reference for this project. The dataset contains order and delivery-related information, including purchase dates, delivery dates, estimated delivery dates, and order status.

### 2. Data Cleaning

The preprocessing process identifies common data quality issues such as:

- Missing values
- Duplicate records
- Incorrect data types
- Inconsistent date formats
- Potential outliers

Appropriate cleaning techniques are applied to improve the reliability of the dataset.

### 3. Feature Engineering

The following logistics features are created:

- **Delivery Time:** Calculated using the purchase date and actual delivery date.
- **Delivery Delay Status:** Identifies whether an order was delivered after the estimated delivery date.
- **Potential Outlier Flag:** Identifies unusually high or low delivery times.

### 4. Outlier Detection

The Interquartile Range (IQR) method is used to identify potential outliers in delivery time data.

### 5. Normalization

Min-Max Scaling is applied to numerical delivery-time data to prepare it for future machine learning algorithms.

## Technologies Used

- Python
- Pandas
- Scikit-learn

## File

- `week2_logistics_preprocessing.py` – Python script demonstrating the Week 2 data cleaning and preprocessing pipeline.

## Expected Outcome

The result of this stage is a clean and structured logistics dataset suitable for exploratory data analysis, KPI calculation, visualization, and predictive modeling in the following stages of the project.
