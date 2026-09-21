import pandas as pd

# Load the dataset
df = pd.read_csv("IMDB Dataset.csv")

# Take a small random sample so we're not processing all 50,000 rows yet
sample = df.sample(100, random_state=42)

# Preview it
print(sample.head())
print(f"\nTotal rows in full dataset: {len(df)}")
print(f"Sample size: {len(sample)}")