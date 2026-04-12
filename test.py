import pandas as pd
df = pd.read_csv("labeled_dataset.csv")
print(df["vibration_x"].describe())
