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



# Week 3: Advanced Data Analysis and Visualization

## Project Overview

Week 3 focuses on performing Exploratory Data Analysis (EDA) and creating visualizations to understand logistics performance. A hypothetical logistics dataset is used to analyze delivery times, shipment volumes, transportation costs, distances, regions, and delivery status.

The purpose of this analysis is to identify operational patterns, potential bottlenecks, cost drivers, and factors that can affect logistics efficiency.

## Objectives

- Perform Exploratory Data Analysis (EDA)
- Calculate descriptive statistics and central tendencies
- Analyze relationships between logistics variables
- Examine correlations between numerical variables
- Analyze transportation costs and delivery times
- Compare logistics performance across different regions
- Identify potential operational bottlenecks
- Create visualizations to communicate insights clearly

## Dataset Variables

The hypothetical logistics dataset contains the following variables:

- `shipment_id` – Unique shipment identifier
- `region` – Operational region
- `transport_mode` – Mode of transportation
- `shipment_volume` – Volume of the shipment
- `distance_km` – Distance travelled
- `delivery_time_days` – Time taken for delivery
- `transport_cost` – Cost of transportation
- `delivery_status` – On-Time or Delayed

## Exploratory Data Analysis

The analysis includes:

- Mean and median calculations
- Standard deviation
- Minimum and maximum values
- Distribution analysis
- Group-based analysis
- Correlation analysis

These techniques help understand the overall characteristics of logistics operations and identify relationships between important performance variables.

## Visualizations

The following visualizations are created using Python and Matplotlib:

### 1. Delivery Time Distribution

A histogram is used to examine the distribution of delivery times and identify common delivery ranges and potential delays.

### 2. Transportation Cost by Transport Mode

A bar chart compares average transportation costs across different transport modes.

### 3. Distance vs Transportation Cost

A scatter plot is used to examine the relationship between shipment distance and transportation cost.

### 4. Average Delivery Time by Region

A bar chart compares delivery performance across different operational regions.

### 5. Delivery Status Analysis

A chart compares the number of On-Time and Delayed shipments to evaluate overall delivery performance.

## Key Insights

The analysis helps identify:

- Factors that may influence transportation costs
- The relationship between distance and logistics expenses
- Regional differences in delivery performance
- Potential delivery bottlenecks
- Overall On-Time and Delayed shipment performance

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib

## Files

- `week3_logistics_analysis_visualization.py` – Python script for Week 3 analysis and visualizations
- `week3_hypothetical_logistics_data.csv` – Hypothetical logistics dataset used for the analysis

## Expected Outcome

This analysis provides insights into logistics performance and supports data-driven decision-making. The results can be used as a foundation for future KPI monitoring, predictive modeling, delivery delay prediction, and logistics optimization.
