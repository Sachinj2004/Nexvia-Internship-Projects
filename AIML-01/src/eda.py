import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Find the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "creditcard.csv"

# Load the dataset
print("Loading the credit card transaction dataset...")

df = pd.read_csv(DATA_PATH)

# Display basic information
print("\nDataset loaded successfully!")
print("-" * 40)

print("Total transactions:", len(df))
print("Total columns:", len(df.columns))

print("\nFirst 5 transactions:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sum())

# Count legitimate and fraudulent transactions
print("\nTransaction class distribution:")
print(df["Class"].value_counts())

# Calculate percentages
print("\nTransaction percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

print("\nAnalysis completed!")