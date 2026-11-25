"""Polish desktop GUI (Tkinter) for scanning monitor photos, inspecting detections, and exporting CSV."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from monitor_reader.decoder import load_config
from monitor_reader.pipeline import (
    MeasurementRecord,
    export_csv,
    list_image_files,
    process_image,
    save_debug_artifacts,
)


def _bgr_to_photo(img_bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    """Convert a BGR numpy image to a resized ImageTk.PhotoImage preserving aspect ratio."""
    h, w = img_bgr.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class MonitorReaderApp:
    """Main Tkinter desktop application for offline LCD monitor reading."""

    def __init__(self, root: tk.Tk, initial_dir: str | Path | None = None) -> None:
        self.root = root
        self.root.title("Odczyt parametrów z monitora LCD (Offline)")
        self.root.geometry("1220x740")
        self.root.minsize(980, 620)

        self.config: dict[str, Any] = load_config()
        self.records: list[MeasurementRecord] = []
        self._worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._is_scanning = False
        self._cancel_requested = False

        # Keep strong references to displayed PhotoImages
        self._crop_photo: ImageTk.PhotoImage | None = None
        self._lcd_photo: ImageTk.PhotoImage | None = None

        default_dir = Path(initial_dir) if initial_dir else Path.cwd()
        self.var_folder = tk.StringVar(value=str(default_dir))
        self.var_delimiter = tk.StringVar(value=", (standard)")
        self.var_save_debug = tk.BooleanVar(value=False)
        self.var_show_roi = tk.BooleanVar(value=True)
        self.var_show_ink = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Gotowy. Wybierz folder ze zdjęciami i kliknij „Skanuj zdjęcia”.")
        self.var_progress = tk.DoubleVar(value=0.0)

        # Quick-edit field variables
        self.var_edit_time = tk.StringVar()
        self.var_edit_date = tk.StringVar()
        self.var_edit_weekday = tk.StringVar()
        self.var_edit_temp = tk.StringVar()
        self.var_field_confs = tk.StringVar(value="Pewność pól: —")

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        # Top control bar
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Folder zdjęć:").pack(side=tk.LEFT, padx=(0, 6))
        self.entry_folder = ttk.Entry(top, textvariable=self.var_folder, width=46)
        self.entry_folder.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Button(top, text="Wybierz folder...", command=self._choose_folder).pack(side=tk.LEFT, padx=3)
        self.btn_scan = ttk.Button(top, text="Skanuj zdjęcia", command=self.start_scan)
        self.btn_scan.pack(side=tk.LEFT, padx=3)
        self.btn_export = ttk.Button(top, text="Eksportuj CSV...", command=self._export_csv_dialog)
        self.btn_export.pack(side=tk.LEFT, padx=3)

        # Second bar: options & progress
        opts = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        opts.pack(fill=tk.X)

        ttk.Label(opts, text="Separator CSV:").pack(side=tk.LEFT, padx=(0, 4))
        delim_cb = ttk.Combobox(
            opts,
            textvariable=self.var_delimiter,
            values=[", (standard)", "; (Excel PL)"],
            state="readonly",
            width=14,
        )
        delim_cb.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Checkbutton(
            opts,
            text="Zapisz obrazy debugowe (podfolder _debug)",
            variable=self.var_save_debug,
        ).pack(side=tk.LEFT, padx=(0, 16))

        self.progress = ttk.Progressbar(opts, variable=self.var_progress, maximum=100.0, length=200)
        self.progress.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Label(opts, textvariable=self.var_status).pack(side=tk.RIGHT, padx=6)

        # Main split pane
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Left pane: results table
        left = ttk.Frame(paned)
        paned.add(left, weight=3)

        cols = ("file", "ts", "time", "date", "weekday", "temp", "conf", "status")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        headers = [
            ("file", "Plik", 155),
            ("ts", "Czas zdjęcia", 130),
            ("time", "Godzina", 68),
            ("date", "Data", 64),
            ("weekday", "Dzień", 56),
            ("temp", "Temp [°C]", 72),
            ("conf", "Pewność", 68),
            ("status", "Status", 110),
        ]
        for cid, label, width in headers:
            self.tree.heading(cid, text=label)
            anchor = tk.W if cid in ("file", "ts", "status") else tk.CENTER
            self.tree.column(cid, width=width, minwidth=50, anchor=anchor)

        self.tree.tag_configure("ok", background="#eef9ee")
        self.tree.tag_configure("low", background="#fff8e1")
        self.tree.tag_configure("err", background="#fdecea")
        self.tree.tag_configure("edited", background="#e8f1fc")

        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

        # Right pane: visual inspection + manual edit
        right = ttk.Frame(paned, padding=(8, 0, 0, 0))
        paned.add(right, weight=2)

        crop_box = ttk.LabelFrame(right, text="Wykryty monitor na zdjęciu", padding=6)
        crop_box.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.lbl_crop = ttk.Label(crop_box, text="Brak podglądu", anchor=tk.CENTER)
        self.lbl_crop.pack(fill=tk.BOTH, expand=True)

        lcd_box = ttk.LabelFrame(right, text="Wyprostowany ekran LCD (440×200)", padding=6)
        lcd_box.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        toggles = ttk.Frame(lcd_box)
        toggles.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(
            toggles,
            text="Pokaż siatkę pól (ROI)",
            variable=self.var_show_roi,
            command=self._refresh_selected_preview,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(
            toggles,
            text="Pokaż mapę kontrastu (ink)",
            variable=self.var_show_ink,
            command=self._refresh_selected_preview,
        ).pack(side=tk.LEFT)

        self.lbl_lcd = ttk.Label(lcd_box, text="Brak podglądu", anchor=tk.CENTER)
        self.lbl_lcd.pack(fill=tk.BOTH, expand=True)

        edit_box = ttk.LabelFrame(right, text="Korekta / Weryfikacja odczytu", padding=8)
        edit_box.pack(fill=tk.X)

        ttk.Label(edit_box, textvariable=self.var_field_confs).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6)
        )

        ttk.Label(edit_box, text="Godzina:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(edit_box, textvariable=self.var_edit_time, width=10).grid(row=1, column=1, sticky=tk.W, padx=(0, 12))
        ttk.Label(edit_box, text="Data:").grid(row=1, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Entry(edit_box, textvariable=self.var_edit_date, width=10).grid(row=1, column=3, sticky=tk.W)

        ttk.Label(edit_box, text="Dzień tyg.:").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(edit_box, textvariable=self.var_edit_weekday, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=(0, 12), pady=(6, 0)
        )
        ttk.Label(edit_box, text="Temp [°C]:").grid(row=2, column=2, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(edit_box, textvariable=self.var_edit_temp, width=10).grid(row=2, column=3, sticky=tk.W, pady=(6, 0))

        ttk.Button(edit_box, text="Zastosuj poprawkę", command=self.apply_manual_edit).grid(
            row=3, column=0, columnspan=4, sticky=tk.E, pady=(8, 0)
        )

    def _choose_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.var_folder.get() or str(Path.cwd()))
        if chosen:
            self.var_folder.set(chosen)

    def _get_selected_delimiter(self) -> str:
        return ";" if self.var_delimiter.get().startswith(";") else ","

    def start_scan(self) -> None:
        if self._is_scanning:
            return
        folder = Path(self.var_folder.get().strip())
        try:
            files = list_image_files(folder)
        except FileNotFoundError as exc:
            messagebox.showerror("Błąd folderu", str(exc))
            return

        if not files:
            messagebox.showinfo("Brak zdjęć", f"Nie znaleziono obsługiwanych zdjęć w:\n{folder}")
            return

        self._is_scanning = True
        self._cancel_requested = False
        self.btn_scan.configure(state=tk.DISABLED)
        self.btn_export.configure(state=tk.DISABLED)
        self.load_records([])
        self.var_progress.set(0.0)
        self.var_status.set(f"Skanowanie 0/{len(files)}...")

        debug_dir = folder / "_debug" if self.var_save_debug.get() else None
        worker = threading.Thread(target=self._scan_worker, args=(files, debug_dir), daemon=True)
        worker.start()
        self.root.after(60, self._poll_worker)

    def _scan_worker(self, files: list[Path], debug_dir: Path | None) -> None:
        total = len(files)
        for idx, path in enumerate(files, start=1):
            if self._cancel_requested:
                break
            rec = process_image(path, self.config)
            if debug_dir is not None:
                save_debug_artifacts(rec, debug_dir)
            self._worker_queue.put(("record", (idx, total, rec)))
        self._worker_queue.put(("done", total))

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self._worker_queue.get_nowait()
                if kind == "record":
                    idx, total, rec = payload
                    self._append_record(rec)
                    self.var_progress.set(100.0 * idx / max(1, total))
                    self.var_status.set(f"Skanowanie {idx}/{total}: {rec.filename}")
                elif kind == "done":
                    total = payload
                    self._is_scanning = False
                    self.btn_scan.configure(state=tk.NORMAL)
                    self.btn_export.configure(state=tk.NORMAL)
                    ok_cnt = sum(1 for r in self.records if r.status == "OK")
                    self.var_status.set(f"Zakończono ({total} zdjęć, OK: {ok_cnt}).")
                    if self.records and not self.tree.selection():
                        first_iid = self.tree.get_children()[0]
                        self.tree.selection_set(first_iid)
                        self._on_select_row()
                    return
        except queue.Empty:
            pass
        if self._is_scanning:
            self.root.after(60, self._poll_worker)

    def _row_tag(self, status: str) -> str:
        if status == "OK":
            return "ok"
        if status == "POPRAWIONO":
            return "edited"
        if status == "NISKA PEWNOŚĆ":
            return "low"
        return "err"

    def _append_record(self, rec: MeasurementRecord) -> None:
        idx = len(self.records)
        self.records.append(rec)
        self.tree.insert(
            "",
            tk.END,
            iid=str(idx),
            values=(
                rec.filename,
                rec.photo_timestamp,
                rec.time or "—",
                rec.date or "—",
                rec.weekday or "—",
                rec.temperature or "—",
                f"{rec.confidence:.2f}",
                rec.status,
            ),
            tags=(self._row_tag(rec.status),),
        )

    def load_records(self, records: list[MeasurementRecord]) -> None:
        """Replace displayed records (used by scanner and tests)."""
        self.records = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        for rec in records:
            self._append_record(rec)
        if self.records:
            first_iid = self.tree.get_children()[0]
            self.tree.selection_set(first_iid)
            self._on_select_row()
        else:
            self._clear_preview()

    def _clear_preview(self) -> None:
        self._crop_photo = None
        self._lcd_photo = None
        self.lbl_crop.configure(image="", text="Brak podglądu")
        self.lbl_lcd.configure(image="", text="Brak podglądu")
        self.var_field_confs.set("Pewność pól: —")
        self.var_edit_time.set("")
        self.var_edit_date.set("")
        self.var_edit_weekday.set("")
        self.var_edit_temp.set("")

    def _get_selected_index(self) -> int | None:
        sel = self.tree.selection()
        if not sel:
            return None
        idx = int(sel[0])
        if 0 <= idx < len(self.records):
            return idx
        return None

    def _on_select_row(self, _event: object | None = None) -> None:
        idx = self._get_selected_index()
        if idx is None:
            self._clear_preview()
            return
        rec = self.records[idx]
        self.var_edit_time.set(rec.time)
        self.var_edit_date.set(rec.date)
        self.var_edit_weekday.set(rec.weekday)
        self.var_edit_temp.set(rec.temperature)
        self.var_field_confs.set(
            f"Pewność pól — godz.: {rec.conf_time:.2f}, data: {rec.conf_date:.2f}, "
            f"dzień: {rec.conf_weekday:.2f}, temp.: {rec.conf_temperature:.2f}"
        )
        self._refresh_selected_preview()

    def _refresh_selected_preview(self) -> None:
        idx = self._get_selected_index()
        if idx is None:
            return
        rec = self.records[idx]

        if rec.crop_preview is not None:
            self._crop_photo = _bgr_to_photo(rec.crop_preview, 420, 210)
            self.lbl_crop.configure(image=self._crop_photo, text="")
        else:
            self._crop_photo = None
            self.lbl_crop.configure(image="", text=rec.error_reason or "Nie wykryto monitora")

        lcd_img = None
        if self.var_show_ink.get() and rec.ink_vis is not None:
            lcd_img = rec.ink_vis
        elif self.var_show_roi.get() and rec.roi_overlay is not None:
            lcd_img = rec.roi_overlay
        elif rec.warped_lcd is not None:
            lcd_img = rec.warped_lcd

        if lcd_img is not None:
            self._lcd_photo = _bgr_to_photo(lcd_img, 420, 195)
            self.lbl_lcd.configure(image=self._lcd_photo, text="")
        else:
            self._lcd_photo = None
            self.lbl_lcd.configure(image="", text="Brak wyciętego ekranu LCD")

    def apply_manual_edit(self) -> bool:
        """Save edited values from the quick-edit inputs back into the selected record."""
        idx = self._get_selected_index()
        if idx is None:
            return False
        rec = self.records[idx]
        rec.time = self.var_edit_time.get().strip()
        rec.date = self.var_edit_date.get().strip()
        rec.weekday = self.var_edit_weekday.get().strip().upper()
        rec.temperature = self.var_edit_temp.get().strip()
        rec.status = "POPRAWIONO"

        self.tree.item(
            str(idx),
            values=(
                rec.filename,
                rec.photo_timestamp,
                rec.time or "—",
                rec.date or "—",
                rec.weekday or "—",
                rec.temperature or "—",
                f"{rec.confidence:.2f}",
                rec.status,
            ),
            tags=(self._row_tag(rec.status),),
        )
        return True

    def export_to_path(self, output_path: str | Path) -> None:
        """Export current records using the selected CSV delimiter."""
        export_csv(self.records, output_path, delimiter=self._get_selected_delimiter())

    def _export_csv_dialog(self) -> None:
        if not self.records:
            messagebox.showinfo("Brak danych", "Najpierw przeskanuj zdjęcia, aby zapisać wyniki.")
            return
        initial_dir = self.var_folder.get() or str(Path.cwd())
        out_path = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile="pomiary_monitor.csv",
            defaultextension=".csv",
            filetypes=[("Plik CSV", "*.csv"), ("Wszystkie pliki", "*.*")],
        )
        if not out_path:
            return
        self.export_to_path(out_path)
        self.var_status.set(f"Wyeksportowano CSV: {Path(out_path).name}")
        messagebox.showinfo("Eksport zakończony", f"Zapisano {len(self.records)} wierszy do:\n{out_path}")


def run_gui(initial_dir: str | Path | None = None) -> None:
    """Start the Tkinter desktop application."""
    root = tk.Tk()
    MonitorReaderApp(root, initial_dir=initial_dir)
    root.mainloop()
