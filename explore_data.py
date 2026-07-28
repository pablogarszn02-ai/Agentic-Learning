import pandas as pd

DATA_PATH = "Walmart_data"

train = pd.read_csv(f"{DATA_PATH}/train.csv")
features = pd.read_csv(f"{DATA_PATH}/features.csv")
stores = pd.read_csv(f"{DATA_PATH}/stores.csv")

print("=== TRAIN.CSV ===")
print(train.shape)
print(train.head())
print(train.columns.tolist())

print("\n=== FEATURES.CSV ===")
print(features.shape)
print(features.head())
print(features.columns.tolist())

print("\n=== STORES.CSV ===")
print(stores.shape)
print(stores.head())
print(stores.columns.tolist())

# Unimos train + features (por Store y Date)
merged = train.merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))

# Unimos el resultado + stores (por Store)
merged = merged.merge(stores, on="Store", how="left")
merged = merged.drop(columns=["IsHoliday_feat"])

print("\n=== DATASET UNIDO ===")
print(merged.shape)
print(merged.head())
print(merged.columns.tolist())

merged.to_csv("Walmart_data/merged_dataset.csv", index=False)
print("\n✅ Dataset unido guardado en Walmart_data/merged_dataset.csv")