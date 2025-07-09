from pathlib import Path
import pandas as pd

DURATION_CSV = "/cpfs02/shared/speechllm/vggsound/duration.csv"
CHUNK_SIZE = 25000
OUTPUT_DIR = Path("data/vggsound/csv_parts/")

df = pd.read_csv(DURATION_CSV, sep="\t")
df = df[(df["duration"] > 0.5) & (df["duration"] < 10.5)]
total_rows = len(df)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for chunk_idx, chunk_start in enumerate(range(0, total_rows, CHUNK_SIZE)):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
    chunk_df = df.iloc[chunk_start:chunk_end]
    chunk_df.to_csv(
        OUTPUT_DIR / f"part_{chunk_idx+1:02d}.csv", index=False, sep="\t"
    )
