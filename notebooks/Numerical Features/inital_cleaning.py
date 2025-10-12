import pandas as pd

csv = pd.read_csv(r"notebooks\Numerical Features\17k_products_amazon_data.csv")

print(csv.head())

print(csv.columns)

df = pd.DataFrame(csv, columns=['bsr_best','sd_price',

       'sd_list_price', 'sd_previous_price', 'sd_number_bought_past_month',  'sd_average_rating', 'sd_total_reviews', 'sd_ratings_count', 'sd_stars',

       'sd_rating_pct_1', 'sd_rating_pct_2', 'sd_rating_pct_3',

       'sd_rating_pct_4', 'sd_rating_pct_5'])

print(f"Columns Names: {df.columns}")

print(f"Summary: {df.info()}")

# identify outliers 

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

upper_bounds = Q3 + 1.5 * IQR

lower_bounds = Q1 - 1.5 * IQR

outliers = df[(df > upper_bounds) | (df < lower_bounds)]

print(f"Outliers")

print(f"First Five Rows: {outliers.head()}")

print(f"Statistics: {outliers.describe()}")

print(f"Summary: {outliers.info()}")

print(f"Shape: {outliers.shape}")

# remove outliers 

remove_outliers = df[(df <= upper_bounds) & (df >= lower_bounds)]

print("remove outliers")

print(f"First Five Rows: {remove_outliers.head()}")

print(f"Statistics: {remove_outliers.describe()}")

print(f"Summary: {remove_outliers.info()}")

print(f"Shape: {remove_outliers.shape}")

# new df called df_clean

df_clean = remove_outliers.dropna(axis=1, how='all') 

df_clean = remove_outliers.drop_duplicates()

# save df clean to a new csv file named clean_data.csv

df_clean.to_csv("clean_data.csv")

print("clean_data")

print(f"First Five Rows: {df_clean.head()}")

print(f"Statistics: {df_clean.describe()}")

print(f"Summary: {df_clean.info()}")

print(f"Shape: {df_clean.shape}")

# Distribution Analysis

# histogram of prices (sd_price)

# histogram of best seller rank (bsr_best)

# histogram of average ratings (sd_average_rating)

# histogram of total reviews (sd_total_reviews)

# histogram of units bought past month (sd_number_bought_past_month)



# Correlation Analysis

# correlation heatmap of all numerical variables



# Relationship Analysis

# scatter plot: price vs number bought past month

# scatter plot: average rating vs total reviews

# scatter plot: price vs average rating

# scatter plot: bsr_best vs number bought past month

# scatter plot: price vs bsr_best (lower rank = better)



# Comparison Analysis

# box plot comparing all price columns (sd_price, sd_list_price, sd_previous_price)

# box plot of rating distribution percentages (pct_1 through pct_5)



# Rating Distribution

# stacked bar chart or grouped bar chart of rating percentages (1-5 stars)

# pie chart of average rating distribution across products



# Statistical Summary

# bar chart of mean values for key metrics

# bar chart showing missing data counts per column





