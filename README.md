# Netflix Content Analysis and Future Trend Prediction Using Time Series and Machine Learning

![Netflix_2015_logo](https://github.com/user-attachments/assets/0bdd05b5-389b-46aa-8728-10cd6819c871)<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="276.742" viewBox="0 0 1024 276.742"><path d="M140.803 258.904c-15.404 2.705-31.079 3.516-47.294 5.676l-49.458-144.856v151.073c-15.404 1.621-29.457 3.783-44.051 5.945v-276.742h41.08l56.212 157.021v-157.021h43.511v258.904zm85.131-157.558c16.757 0 42.431-.811 57.835-.811v43.24c-19.189 0-41.619 0-57.835.811v64.322c25.405-1.621 50.809-3.785 76.482-4.596v41.617l-119.724 9.461v-255.39h119.724v43.241h-76.482v58.105zm237.284-58.104h-44.862v198.908c-14.594 0-29.188 0-43.239.539v-199.447h-44.862v-43.242h132.965l-.002 43.242zm70.266 55.132h59.187v43.24h-59.187v98.104h-42.433v-239.718h120.808v43.241h-78.375v55.133zm148.641 103.507c24.594.539 49.456 2.434 73.51 3.783v42.701c-38.646-2.434-77.293-4.863-116.75-5.676v-242.689h43.24v201.881zm109.994 49.457c13.783.812 28.377 1.623 42.43 3.242v-254.58h-42.43v251.338zm231.881-251.338l-54.863 131.615 54.863 145.127c-16.217-2.162-32.432-5.135-48.648-7.838l-31.078-79.994-31.617 73.51c-15.678-2.705-30.812-3.516-46.484-5.678l55.672-126.75-50.269-129.992h46.482l28.377 72.699 30.27-72.699h47.295z" fill="#d81f26"/></svg>

# Introduction
This project uses a variety of data science tools to investigate, analyze, and forecast trends in Netflix content. 
To glean information about content categories, directors, genres, and national contributions to Netflix's library, the dataset is processed and examined. 
Time series forecasting is often used to project future patterns in content. Additionally, possible recommendation systems and trend forecasting make use 
of machine learning models.

# Key Objectives:
- Explore and visualize Netflix's content distribution (genres, release years, countries, etc.).
- Build a content-based recommendation system.
- Predict future trends in Netflix content using time series analysis.
- Train machine learning models for content-based predictions.

# Table of Contents
- Installation
- Features
- Exploratory Data Analysis (EDA)
- Time Series Prediction
- Machine Learning Models
- Recommendation System
- Usage
- Results
- Contributing
- License

# Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AnirudhyaDas/Netflix-Content-Analysis-and-Future-Trend-Prediction.git

2. Navigate to the project folder:
   ```bash
   cd Netflix-Content-Analysis-and-Future-Trend-Prediction
   
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   
5. Open the Netflix.ipynb notebook to explore the analysis.

# Features
### 1. Data Cleaning and Feature Engineering
- Missing value handling: Handle missing entries in key columns.
- Feature extraction: Created features such as year_added, month_added, duration_minutes, etc.
- Content categorization: Breakdown content into genres, countries, and other relevant factors.

### 2. Exploratory Data Analysis (EDA)
- Visualizations for key metrics like:
   - Top 10 directors with the most titles.
   - Distribution of genres and their popularity.
   - Country-wise content contributions.
   - Word cloud of movie titles to find frequent keywords.

### 3. Time Series Trend Prediction
- Applied SARIMA (Seasonal ARIMA) model to forecast Netflix content releases for future years.
- Used Exponential Smoothing to predict trends in content types over time.
- Evaluated model performance using metrics like Mean Squared Error (MSE).

### 4. Machine Learning Models
- Built Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest models to predict content types.
- Performed hyperparameter tuning using GridSearchCV for optimized results.
- Evaluated model performance using accuracy, precision, and recall metrics.

### 5. Content-Based Recommendation System
- Implemented a content-based filtering recommendation system.
- Suggested similar titles based on user-selected attributes like genre and director.

# Exploratory Data Analysis (EDA)
### Visualizations:
- Top 10 Directors: A bar chart showcasing the directors with the most titles.
- Content Breakdown by Genre: Pie chart visualizing the distribution of genres in Netflix’s catalog.
- Content Over Time: A line plot showing the number of content additions over the years.

# Feature Engineering
### Steps Taken:
- Date Parsing: Converted date_added to datetime format for time-based analysis.
- Duration Extraction: For movies, extracted minutes from the duration column.
- One-Hot Encoding: Created dummy variables for categorical columns such as genres.
 ```python
   df['date_added'] = pd.to_datetime(df['date_added'])
   df['duration_minutes'] = df['duration'].apply(lambda x: extract_minutes(x))

# Time Series Prediction
### SARIMA (Seasonal ARIMA):
We used SARIMA to forecast future content releases on Netflix. This model captures both the seasonal patterns and overall trends in the dataset.
 ```python
   from statsmodels.tsa.statespace.sarimax import SARIMAX
   sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
   sarima_fit = sarima_model.fit()

