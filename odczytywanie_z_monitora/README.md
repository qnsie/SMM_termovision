# Odczyt parametrów z monitora LCD (Offline)

Lokalna aplikacja desktopowa (Python + OpenCV + Tkinter) do automatycznego wykrywania małego białego zegara/termometru LCD na zdjęciach, prostowania perspektywy, odczytu wyświetlanych parametrów **100% offline** (bez zewnętrznych API) oraz eksportu pomiarów do pliku CSV.

## Odczytywane parametry
- **Godzina** (`HH:MM`) — górny wiersz dużych cyfr 7-segmentowych (działa zarówno przy zapalonym, jak i wygaszonym dwukropku)
- **Data** (`MM-DD`) — lewy dolny obszar ekranu
- **Dzień tygodnia** (`MON`..`SUN`) — środkowy dolny obszar ekranu
- **Temperatura** (`°C`) — prawy dolny obszar ekranu
- **Czas wykonania zdjęcia** — pobierany z metadanych EXIF (`DateTimeOriginal`), nazwy pliku (`YYYYMMDD_HHMMSS`) lub daty modyfikacji pliku

## Uruchomienie aplikacji okienkowej (GUI)

Najszybciej przez plik `uruchom.bat` (dwuklik) albo z terminala:

```powershell
.\.venv\Scripts\python.exe app.py
```

W oknie programu:
1. Wybierz folder ze zdjęciami (`Wybierz folder...`).
2. Kliknij `Skanuj zdjęcia` — w tabeli pojawią się odczyty i wskaźnik pewności.
3. Kliknij dowolny wiersz, aby zobaczyć wykryty monitor oraz wyprostowany ekran LCD (z opcjonalną siatką pól ROI lub mapą kontrastu). W razie potrzeby możesz ręcznie poprawić odczyt przed eksportem.
4. Wybierz separator CSV (`,` lub `;` dla polskiego Excela) i kliknij `Eksportuj CSV...`.

## Uruchomienie z linii poleceń (CLI)

```powershell
.\.venv\Scripts\python.exe cli.py --input . --output wyniki.csv --delimiter ";" --debug-dir debug_out
```

Parametry:
- `--input`, `-i` — folder ze zdjęciami lub pojedynczy plik
- `--output`, `-o` — ścieżka pliku wynikowego CSV (domyślnie `wyniki.csv`, kodowanie `UTF-8-BOM`)
- `--delimiter`, `-d` — separator kolumn `,` lub `;`
- `--debug-dir` — opcjonalny folder zapisu obrazów diagnostycznych (`*_detect.png`, `*_lcd_roi.png`, `*_ink.png`)
- `--config` — opcjonalna ścieżka do własnego pliku `config.json`

## Format pliku CSV

Kolumny wyjściowe:
`plik,czas_zdjecia,godzina,data,dzien_tygodnia,temperatura_C,pewnosc,status`

## Kalibracja (`config.json`)

Współrzędne pól na wyprostowanym ekranie (`440×200`), grubość segmentów, pochylenie czcionki i próg statusu `NISKA PEWNOŚĆ` znajdują się w pliku `config.json`.

## Testy automatyczne

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
