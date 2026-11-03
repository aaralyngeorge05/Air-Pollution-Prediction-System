
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

CSV_FILE = "city_day.csv"
DATABASE_NAME = "AirPollutionDB"
COLLECTION_NAME = "historical_air_quality"
BATCH_SIZE = 1000

def main():
    print("Connecting to MongoDB...")

    client = MongoClient(
        "mongodb://localhost:27017/",
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")

    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    print("Reading historical air-quality dataset...")
    df = pd.read_csv(CSV_FILE)

    # Convert missing values to None so MongoDB can store them cleanly.
    df = df.astype(object).where(pd.notna(df), None)

    # Convert each record's date to a Python datetime.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        )
        df["Date"] = df["Date"].apply(
            lambda value: value.to_pydatetime()
            if pd.notna(value) else None
        )

    records = df.to_dict(orient="records")
    total = len(records)

    print(f"Dataset contains {total:,} records.")

    existing = collection.count_documents({})
    if existing > 0:
        print(
            f"The collection already contains {existing:,} records."
        )
        print("No data imported, to avoid duplicate imports.")
        client.close()
        return

    inserted = 0

    for start in range(0, total, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        collection.insert_many(batch)
        inserted += len(batch)
        print(f"Imported {inserted:,} of {total:,} records...")

    print("\nImport completed successfully!")
    print("Database:", DATABASE_NAME)
    print("Collection:", COLLECTION_NAME)
    print("Documents stored:", collection.count_documents({}))

    client.close()

if __name__ == "__main__":
    main()
