import argparse
from pathlib import Path
import numpy as np

from configs import settings
from src.ir_video_tracker import (
    DEFAULT_CSV_OUTPUT_DIR,
    DEFAULT_VIDEO_OUTPUT_DIR,
    analyze_ir_video,
)


def _format_temp_stats(series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "brak danych"
    return (
        f"min={valid.min():.2f} C | srednia={valid.mean():.2f} C | "
        f"mediana={valid.median():.2f} C | max={valid.max():.2f} C (n={len(valid)})"
    )


def main() -> None:
    default_video = settings.BASE_DIR / "data" / "test" / "Testy wtorek" / "FLIR" / "FLIR2333.mp4"

    parser = argparse.ArgumentParser(
        description="Analiza nagran termowizyjnych FLIR T420 (.mp4): detekcja twarzy, OCR skali i pomiar temperatur.",
    )
    parser.add_argument("--video", type=Path, default=default_video, help="Sciezka do pliku .mp4")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT_DIR, help="Katalog wyjsciowy na wideo .mp4")
    parser.add_argument("--out-csv-dir", type=Path, default=DEFAULT_CSV_OUTPUT_DIR, help="Katalog wyjsciowy na pliki .csv")
    parser.add_argument("--out-video", type=Path, default=None, help="Sciezka wyjsciowego wideo .mp4")
    parser.add_argument("--out-csv", type=Path, default=None, help="Sciezka wyjsciowego pliku .csv")
    parser.add_argument(
        "--manual-range",
        type=float,
        nargs=2,
        metavar=("T_MIN", "T_MAX"),
        default=None,
        help="Sztywny zakres temperatur [T_MIN T_MAX], wylacza Auto-OCR",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Prog pewnosci YOLO (domyslnie 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozmiar wejscia YOLO (domyslnie 640)")
    parser.add_argument(
        "--preprocess",
        choices=["temp_inferno", "gray_inferno", "raw"],
        default="temp_inferno",
        help="Sposob przygotowania klatki przed YOLO",
    )
    parser.add_argument("--device", default="0", help="Urzadzenie dla YOLO (0 dla GPU lub cpu)")
    parser.add_argument("--ocr-step", type=int, default=5, help="Co ile klatek uruchamiac OCR skali")
    parser.add_argument("--min-eye-width", type=int, default=50, help="Min. szerokosc ramki twarzy [px] dla MediaPipe")
    args = parser.parse_args()

    out_video = args.out_video or (args.out_dir / f"{args.video.stem}_analyzed.mp4")
    out_csv = args.out_csv or (args.out_csv_dir / f"{args.video.stem}_results.csv")
    manual_range = tuple(args.manual_range) if args.manual_range is not None else None
    device: int | str = int(args.device) if str(args.device).isdigit() else str(args.device)

    df, summary = analyze_ir_video(
        video_path=args.video,
        output_video_path=out_video,
        output_csv_path=out_csv,
        manual_temp_range=manual_range,
        yolo_conf=args.conf,
        yolo_imgsz=args.imgsz,
        yolo_preprocess=args.preprocess,
        device=device,
        min_bbox_width_for_landmarks=args.min_eye_width,
        ocr_every_n_frames=args.ocr_step,
    )

    print("\n=== PODSUMOWANIE ANALIZY WIDEO IR ===")
    print(f"Wideo wejsciowe:       {summary['video_path']}")
    print(f"Wideo wynikowe:        {summary['output_video_path']}")
    print(f"Plik CSV:              {summary['output_csv_path']}")
    print(f"Tryb skali:            {summary['scale_mode']}")
    print(f"Urzadzenie YOLO:       {summary['device']} (preprocess={args.preprocess}, conf={args.conf})")
    print(f"Liczba klatek:         {summary['total_frames']}")
    print(f"Detekcja paska barw:   {summary['colorbar_detected_frames']}/{summary['total_frames']}")
    if summary["scale_mode"] == "Auto-OCR":
        print(f"Poprawne odczyty OCR:  {summary['ocr_valid_reads']}/{summary['ocr_attempts']}")
        print(
            f"Zakres skali z OCR:    scale_min=[{df['scale_min'].min():.1f}..{df['scale_min'].max():.1f}] C, "
            f"scale_max=[{df['scale_max'].min():.1f}..{df['scale_max'].max():.1f}] C"
        )
    print(
        f"Skutecznosc YOLO:      {summary['yolo_detected_frames']}/{summary['total_frames']} "
        f"({summary['yolo_detection_rate_pct']:.1f}%)"
    )
    print(
        f"Skutecznosc MediaPipe: {summary['mp_detected_frames']}/{summary['total_frames']} "
        f"({summary['mp_detection_rate_pct']:.1f}% wszystkich klatek | "
        f"{summary['mp_detected_frames']}/{summary['mp_eligible_frames']} = "
        f"{summary['mp_eligible_rate_pct']:.1f}% przy bbox >= {args.min_eye_width} px)"
    )
    print(f"T_head_max:            {_format_temp_stats(df['T_head_max'])}")
    print(f"T_eye_inner_max:       {_format_temp_stats(df['T_eye_inner_max'])}")
    print(f"Czas wykonania:        {summary['elapsed_s']:.2f} s\n")


if __name__ == "__main__":
    main()
