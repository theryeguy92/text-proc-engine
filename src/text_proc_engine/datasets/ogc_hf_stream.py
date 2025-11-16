from datasets import load_dataset

ds_sample = load_dataset(
    "racineai/OGC_MEGA_MultiDomain_DocRetrieval",
    split="train",
    streaming=True # don't be link me... this ALWAYS NEEDS TO BE TRUE... my ssd is composed of only parquet files now.
)

print("Pulling dataset object:", ds_sample)

# print column names/check out schema
print("column names:", ds_sample.column_names)

# Important note: When streaming datasets, indexing data like ds_sample[0], does not work.
# We need to iterate, be more specific
print("\nFirst 3 rows:\n")
for i, row in enumerate(ds_sample):
    print(f"Row {i}:")
    for k, v in row.items():
        # So we don't print massive blobs
        if isinstance(v, str) and len(v) > 200:
            v_preview = v[:200] + "..."
        else:
            v_preview = v
        print(f" {k}:{v_preview}")
    print()

    if i >= 2:
        break