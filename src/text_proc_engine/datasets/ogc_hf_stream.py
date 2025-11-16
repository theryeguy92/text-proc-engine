from datasets import load_dataset

def debug_print_first_rows(n: int = 3):
    ds_sample = load_dataset(
        "racineai/OGC_MEGA_MultiDomain_DocRetrieval",
        split="train",
        streaming=True,
    )
    print("Pulling dataset object:", ds_sample)
    print("column names:", ds_sample.column_names)

    print(f"\nFirst {n} rows:\n")
    for i, row in enumerate(ds_sample):
        print(f"Row {i}:")
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 200:
                v_preview = v[:200] + "..."
            else:
                v_preview = v
            print(f" {k}: {v_preview}")
        print()
        if i >= n - 1:
            break

if __name__ == "__main__":
    debug_print_first_rows(3)
