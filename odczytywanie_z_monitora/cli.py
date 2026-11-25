"""Command-line interface for batch reading monitor photos into CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from monitor_reader.decoder import load_config
from monitor_reader.pipeline import (
    MeasurementRecord,
    export_csv,
    list_image_files,
    process_image,
    save_debug_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline odczyt parametrów z monitora LCD na zdjęciach do pliku CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Ścieżka do folderu ze zdjęciami lub pojedynczego pliku.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="wyniki.csv",
        help="Ścieżka wyjściowego pliku CSV (domyślnie: wyniki.csv).",
    )
    parser.add_argument(
        "--delimiter",
        "-d",
        default=",",
        choices=[",", ";"],
        help="Separator kolumn CSV (',' lub ';').",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Opcjonalny folder na obrazy diagnostyczne (wykrycie, ROI, mapa kontrastu).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Opcjonalna ścieżka do pliku config.json.",
    )
    args = parser.parse_args(argv)

    files = list_image_files(args.input)
    if not files:
        print(f"Nie znaleziono obsługiwanych zdjęć w: {args.input}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    records: list[MeasurementRecord] = []
    for path in files:
        rec = process_image(path, cfg)
        records.append(rec)
        if args.debug_dir:
            save_debug_artifacts(rec, Path(args.debug_dir))
        print(
            f"{rec.filename:24s} | {rec.photo_timestamp} | "
            f"{rec.time or '--:--':>5s} | {rec.date or '--.--':>5s} | "
            f"{rec.weekday or '---':>3s} | {rec.temperature or '--':>3s}C | "
            f"conf={rec.confidence:.3f} | {rec.status}"
        )

    export_csv(records, args.output, delimiter=args.delimiter)
    print(f"\nZapisano {len(records)} rekordów do: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
