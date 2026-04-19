import pandas as pd

# Convert dev set
dev = pd.read_csv("SplitData/dev_set.csv")
dev.to_excel("SplitData/dev_set.xlsx", index=False)

# Convert test set
test = pd.read_csv("SplitData/test_set.csv")
test.to_excel("SplitData/test_set.xlsx", index=False)

print("✅ Conversion complete!")
