import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Specify the location of the dataset
dataset_location = 'C:/Users/suhai/IMDb Movies India.csv'

# Load data from the CSV file
movie_df = pd.read_csv(dataset_location, encoding='cp1252')

# Preview the first few entries of the dataset
print(movie_df.head())

# Check for any missing values in the dataset
print(movie_df.isnull().sum())

# Remove rows where the 'rating' column has missing values
movie_df = movie_df.dropna(subset=['rating'])

# Fill missing values in other columns using forward fill method
movie_df.fillna(method='ffill', inplace=True)

# Convert categorical columns into numerical format using one-hot encoding
movie_df_encoded = pd.get_dummies(movie_df, columns=['genre', 'director', 'actors'], drop_first=True)

# Define feature matrix and target vector
X = movie_df_encoded.drop('rating', axis=1)
y = movie_df_encoded['rating']

# Split the data into training and testing sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_set, y_train_set)

# Predict the ratings for the test dataset
predictions = lin_reg_model.predict(X_test_set)

# Compute performance metrics
mse_value = mean_squared_error(y_test_set, predictions)
r2_value = r2_score(y_test_set, predictions)

print(f"Mean Squared Error: {mse_value}")
print(f"R2 Score: {r2_value}")
