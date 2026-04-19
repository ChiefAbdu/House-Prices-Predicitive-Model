import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
housing = pd.read_csv("california_housing_dataset/california_housing.csv")

# Step 1: Create income categories
housing["income_cat"] = pd.cut(
    housing["Median_Income_10kUSD"],
    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Step 2: Stratified split
dev_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42,
    stratify=housing["income_cat"]
)

# Step 3: Cleanup
for set_ in (dev_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print(f"Development Set size: {len(dev_set)}")
print(f"Test Set size: {len(test_set)}")

# Save splits to files
dev_set.to_csv("dev_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)

print("✅ Files saved:")
print("dev_set.csv")
print("test_set.csv")
