import pandas as pd
from pathlib import Path
import sys

data_dir = Path(__file__).parent / "data"

xiu_path = data_dir / "datashare.csv"

features = ["mom1m", "bm", "mvel1"]

columns = ["permno", "DATE"] + features

def preprocess_month(month_data):

    features = ["mom1m", "bm", "mvel1"]

    for feature in features:

        month_data[f"{feature}_rank"] = (
            month_data[feature].rank()
        )

        n = month_data[feature].count()

        month_data[f"{feature}_scaled"] = (
            2 * month_data[f"{feature}_rank"] / (n + 1) - 1
        )

    print(
        "Processed month:",
        month_data["DATE"].iloc[0],
        "| Stocks:",
        len(month_data)
    )

    return month_data

month_buffer = pd.DataFrame()

output_path = data_dir / "gkx_features_3char_full.incomplete.csv"

final_output_path = data_dir / "gkx_features_3char_full.csv"

if output_path.exists():
    raise FileExistsError(
        f"Output file already exists: {output_path}"
    )

if final_output_path.exists():
    raise FileExistsError(
        f"Final output file already exists: {final_output_path}"
    )

first_write = True
input_rows = 0
written_rows = 0
input_month_counts = {}

target_dates = [20001031, 20001130]
selected_chunks = []
TEST_MODE = False

for chunk in pd.read_csv(
    xiu_path,
    usecols=columns,
    chunksize=200000
):

    if TEST_MODE:
        selected = chunk[
            chunk["DATE"].isin(target_dates)
        ]

        if not selected.empty:
            selected_chunks.append(selected)

        print(
            "Target rows in this chunk:",
            len(selected)
        )

        if chunk["DATE"].iloc[-1] > max(target_dates):
            print("Finished reading target months")

            target_data = pd.concat(
                selected_chunks,
                ignore_index=True
            )

            print("Total target rows:", len(target_data))

            print("\n=== MONTHLY COUNTS ===")
            print(target_data.groupby("DATE").size())

            assert len(target_data) == 16589

            print("\n=== MONTHLY RANK NORMALIZATION ===")

            for date, month_data in target_data.groupby("DATE"):
                print("\n=== VALID FEATURE COUNTS ===")
                print("Month:", date)
                print(month_data[features].count())

                processed = preprocess_month(month_data.copy())
                test_output_path = data_dir / "gkx_two_months_test.csv"

                processed.to_csv(
                    test_output_path,
                    mode="w" if date == target_dates[0] else "a",
                    header=date == target_dates[0],
                    index=False
                )

                apple = processed[
                    processed["permno"] == 14593
                    ]

                print("\nApple:", date)

                print(
                    apple[
                        [
                            "DATE",
                            "permno",
                            "mom1m_scaled",
                            "bm_scaled",
                            "mvel1_scaled"
                        ]
                    ].to_string(index=False)
                )

            sys.exit(0)

        continue

    input_rows += len(chunk)

    for date, count in chunk.groupby("DATE").size().items():
        input_month_counts[date] = (
                input_month_counts.get(date, 0) + count
        )

    # 1. Add new records to buffer
    month_buffer = pd.concat(
        [month_buffer, chunk],
        ignore_index=True
    )

    # 2. Find the latest month
    latest_date = month_buffer["DATE"].max()

    # 3. Separate completed months
    completed_month = month_buffer[
        month_buffer["DATE"] < latest_date
    ].copy()

    if not completed_month.empty:


        for date, month_data in completed_month.groupby("DATE"):

            processed = preprocess_month(month_data.copy())

            processed.to_csv(
                output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )

            first_write = False
            written_rows += len(processed)


    # 4. Keep the unfinished month
    month_buffer = month_buffer[
        month_buffer["DATE"] == latest_date
    ].copy()

    print("\n=== CHUNK PROCESSING SUMMARY ===")
    print("Completed rows:", len(completed_month))
    print("Buffer rows:", len(month_buffer))

print("\n=== FINAL MONTH AT END OF FILE ===")

if not month_buffer.empty:
    print("Final month rows:", len(month_buffer))

    processed = preprocess_month(month_buffer.copy())
    processed.to_csv(
        output_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

    first_write = False
    written_rows += len(processed)

print("\n=== SAVED DATA VALIDATION ===")

saved_rows = 0

min_date = None
max_date = None
saved_month_counts = {}

saved_buffer = pd.DataFrame()
saved_duplicate_count = 0

for saved_chunk in pd.read_csv(
    output_path,
    usecols=["DATE", "permno"],
    chunksize=200000
):
    saved_rows += len(saved_chunk)

    # Add the new chunk to the validation buffer
    saved_buffer = pd.concat(
        [saved_buffer, saved_chunk],
        ignore_index=True
    )

    # Find the latest month
    latest_saved_date = saved_buffer["DATE"].max()

    # Select completed months
    completed_saved = saved_buffer[
        saved_buffer["DATE"] < latest_saved_date
    ].copy()

    # Check duplicates in completed months
    if not completed_saved.empty:
        saved_duplicate_count += (
            completed_saved.duplicated(
                subset=["permno", "DATE"]
            ).sum()
        )

    # Keep the unfinished month
    saved_buffer = saved_buffer[
        saved_buffer["DATE"] == latest_saved_date
    ].copy()

    for date, count in saved_chunk.groupby("DATE").size().items():
        saved_month_counts[date] = (
            saved_month_counts.get(date, 0) + count
        )

    chunk_min = saved_chunk["DATE"].min()
    chunk_max = saved_chunk["DATE"].max()

    if min_date is None or chunk_min < min_date:
        min_date = chunk_min

    if max_date is None or chunk_max > max_date:
        max_date = chunk_max

# Check duplicates in the final buffered month
if not saved_buffer.empty:
    saved_duplicate_count += (
        saved_buffer.duplicated(
            subset=["permno", "DATE"]
        ).sum()
    )

print("\n=== SAVED DUPLICATE VALIDATION ===")
print("Duplicate rows:", saved_duplicate_count)

assert saved_duplicate_count == 0

print("Saved duplicate check passed!")

print("Input rows:", input_rows)
print("Written rows:", written_rows)
print("Saved rows:", saved_rows)

assert input_rows == written_rows
assert written_rows == saved_rows

print("First DATE:", min_date)
print("Last DATE:", max_date)

assert min_date == 19570131
assert max_date == 20211231

print("\n=== MONTHLY COUNT VALIDATION ===")

assert input_month_counts == saved_month_counts

print("Months checked:", len(input_month_counts))
print("All monthly counts match!")

print("\nAll row count checks passed!")

if final_output_path.exists():
    raise FileExistsError(
        f"Final output file already exists: {final_output_path}"
    )

output_path.rename(final_output_path)

print("Final output saved:", final_output_path)

# print("\n=== MONTHLY COUNTS ===")
# print(saved.groupby("DATE").size())
#
# assert written_rows == len(saved)
#
# print("\n=== PREPROCESSED OCTOBER ===")
#
# october = saved[
#     saved["DATE"] == 20001031
# ]
#
# print(
#     october[
#         ["DATE", "permno", "mom1m",
#          "mom1m_rank", "mom1m_scaled"]
#     ].to_string(index=False)
# )
#
# print("\nAll incremental writing checks passed!")