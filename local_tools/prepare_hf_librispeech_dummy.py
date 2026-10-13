"""把 Hugging Face LibriSpeech dummy Parquet 展开为本地音频目录和 metadata.csv。"""
import argparse
import csv
from pathlib import Path

import pyarrow.parquet as parquet


def prepare(input_parquet: Path, output_dir: Path, max_samples: int) -> int:
    table = parquet.read_table(input_parquet)
    rows = table.slice(0, min(max_samples, table.num_rows)).to_pylist()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["audio_path", "text", "language", "speaker_id", "source_id"],
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            audio = row["audio"]
            filename = f"{index:04d}_{row['id']}.flac"
            (audio_dir / filename).write_bytes(audio["bytes"])
            writer.writerow(
                {
                    "audio_path": f"audio/{filename}",
                    "text": row["text"],
                    "language": "en",
                    "speaker_id": str(row["speaker_id"]),
                    "source_id": row["id"],
                }
            )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="展开 Hugging Face LibriSpeech dummy 数据集。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()
    count = prepare(args.input, args.output, args.max_samples)
    print(f"已展开 {count} 条样本：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
