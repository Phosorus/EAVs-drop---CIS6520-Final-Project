EAVs-drop Final Project - Pipeline & Display

Usage
pwsh run_pipeline.ps1                          # most severe crash (default)

pwsh run_pipeline.ps1 -EventId synshrp2_XXXXXXXX #pick spesific event

pwsh run_pipeline.ps1 -Random                  # random event (testing)

pwsh run_pipeline.ps1 -SkipBuild               # re-plot without rebuilding

You must put the data into the relevant "data" files, in order to be used.

Here is a download link of all of the data files, https://drive.google.com/file/d/1p4DwbPzvmoOzgdv9HEsBS8luR4tQJLqz/view?usp=sharing.
Drag the "data" file into crash_pipeline. 

Test Cases
build_crash_dataset.py offers error cases for failed conversions.
visualize_crash.py uses the created datasets as tests.
Project structure
```
crash\_pipeline/
├── run\_pipeline.ps1              # Windows one-command runner
├── run\_pipeline.sh               # Linux / macOS runner
├── README.md
├── scripts/
│   ├── build\_crash\_dataset.py   # Data pipeline
│   └── visualize\_crash.py       # Interactive 3D visualizer
├── data/                         # put downloaded datasets here
│   ├── synshrp2/
│   │   ├── Tabular\_records.tab  # event annotations
│   │   └── kinematics/          # <Event\_ID>.json files
│   ├── ciss/
│   │   └── EDREVENT.csv         # from NHTSA's bulk CISS table export
│   └── har/
│       └── sensoringData\_acc.csv # from the UDC "data\_raw" release
└── output/                       # generated on first run
```
Datasets USed

SynSHRP2 (crash kinematic time-series)
  URL: https://dataverse.vtti.vt.edu/dataset.xhtml?persistentId=doi:10.15787/VTT1/FOZRSM
  Extract the JSON files from Kinematic_Signals.zip into `data/synshrp2/kinematics/`
  Place `Tabular\_records.tab` directly in `data/synshrp2/`

NHTSA CISS (ground-truth EDR delta-V)
  URL: https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CISS/
  Only one table is used: place EDREVENT.csv in `data/ciss/`
  
Real-Life HAR Dataset (normal driving / phone baseline)
  URL: https://lbd.udc.es/research/real-life-HAR-dataset/
  Place sensoringData_acc.csv directly in `data/har/`
  Note: very large file
  
Output files
  File	Contents
    output/crashes.csv	Crash events — derived delta-V, severity, signal quality (SynSHRP2 + CISS)
    output/normal.csv	Normal driving windows (HAR)
    output/validation.csv	CISS ground-truth EDR delta-V, used only as a comparison reference in the visualizer's "Similar Events" panel — not matched to any specific SynSHRP2/HAR event
    output/timeseries/<id>.npz	Per-event arrays: time, accel, delta_v (SynSHRP2 + HAR only — CISS has no raw signal, only a scalar delta-V, so CISS events can't be plotted directly)
    output/plot_<id>.html	Interactive Plotly HTML, tabbed (Force Trajectory / Vehicle Path / Delta-V Timeline / Similar Events)

Dependencies
  Python 3.9+, auto-installed by runners:
  numpy, pandas, scipy, plotly

Severity thresholds (NHTSA)
  Low:      delta-V < 15 km/h
  Moderate: 15-35 km/h
  Severe:   > 35 km/h
