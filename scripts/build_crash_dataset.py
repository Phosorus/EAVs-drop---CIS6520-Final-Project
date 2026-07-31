"Builds crashes.csv, normal.csv, validation.csv, and timeseries/*.npz from SynSHRP2, CISS, and HAR source data."

import json
import argparse
import logging
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TARGET_HZ       = 100
CLIP_G          = 16.0
G_TO_MS2        = 9.80665
MS2_TO_KMH      = 3.6
WINDOW_PRE_S    = 5.0
WINDOW_CRASH_S  = 0.3
WINDOW_S        = WINDOW_PRE_S + WINDOW_CRASH_S
WINDOW_PRE_MS   = WINDOW_PRE_S * 1000.0
WINDOW_CRASH_MS = WINDOW_CRASH_S * 1000.0
LOWPASS_HZ      = 30.0
HIGHPASS_HZ     = 0.1
SEV_LOW_MAX     = 15.0
SEV_MOD_MAX     = 35.0
MPH_TO_KMH      = 1.609344
PLAUSIBLE_DV_MPH = 150.0
HAR_CHUNK_ROWS  = 2_000_000
HAR_MIN_SAMPLES_PER_WINDOW = 5
HAR_SRC_HZ_MIN, HAR_SRC_HZ_MAX = 1.0, 500.0


def bandpass(data, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [HIGHPASS_HZ / nyq, LOWPASS_HZ / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def resample(signal, src_hz, dst_hz=TARGET_HZ):
    """Linear-interpolation resample; src_hz == dst_hz is a no-op."""
    if src_hz == dst_hz:
        return signal
    n = len(signal)
    if n < 2:
        raise ValueError(f"Cannot resample {n} sample(s)")
    duration = (n - 1) / src_hz
    t = np.arange(n, dtype=float) / src_hz
    n2 = int(round(duration * dst_hz)) + 1
    t2 = np.arange(n2, dtype=float) / dst_hz
    t2 = t2[t2 <= duration + 1e-9]
    return interp1d(t, signal, kind="linear", axis=0,
                    bounds_error=False, fill_value="extrapolate")(t2)


def clip_g(signal):
    return np.clip(signal, -CLIP_G, CLIP_G)


def align_axes(accel_ms2):
    "Rotates phone axes so gravity lands on Z, then removes it."
    gravity      = np.mean(accel_ms2, axis=0)
    gravity_norm = gravity / (np.linalg.norm(gravity) + 1e-9)
    dom          = int(np.argmax(np.abs(gravity_norm)))
    order        = [i for i in range(3) if i != dom] + [dom]
    aligned      = accel_ms2[:, order].copy()
    aligned[:, 2] -= np.sign(gravity_norm[dom]) * G_TO_MS2
    return aligned


def integrate_delta_v(accel_ms2, fs=TARGET_HZ):
    return np.cumsum(accel_ms2) * (1.0 / fs) * MS2_TO_KMH


def classify_severity(dv_kmh):
    if dv_kmh < SEV_LOW_MAX:
        return "low"
    if dv_kmh < SEV_MOD_MAX:
        return "moderate"
    return "severe"


def pad_or_trim(accel, target_samples):
    n = len(accel)
    if n >= target_samples:
        return accel[-target_samples:]
    return np.vstack([np.zeros((target_samples - n, accel.shape[1])), accel])


def summarise(accel, fs=TARGET_HZ):
    "Scalar metrics for a processed window; _dv_series is popped by the caller."
    dv_long = integrate_delta_v(accel[:, 0], fs)
    dv_lat  = integrate_delta_v(accel[:, 1], fs)
    dv_long_max = float(np.max(np.abs(dv_long)))
    dv_lat_max  = float(np.max(np.abs(dv_lat)))
    dv_res      = np.hypot(dv_long_max, dv_lat_max)
    clip_ratio  = float(np.mean(np.abs(accel / G_TO_MS2) >= CLIP_G))
    return {
        "delta_v_longitudinal": round(dv_long_max, 3),
        "delta_v_lateral"     : round(dv_lat_max, 3),
        "delta_v_resultant"   : round(dv_res, 3),
        "severity_label"      : classify_severity(dv_res),
        "peak_accel_g"        : round(float(np.max(np.abs(accel / G_TO_MS2))), 3),
        "clipped_ratio"       : round(clip_ratio, 4),
        "window_duration_s"   : round(len(accel) / fs, 2),
        "sampling_rate_hz"    : int(fs),
        "_dv_series"          : dv_long,
    }


def save_timeseries(event_id, accel, fs, ts_dir, dv_series):
    ts_dir.mkdir(parents=True, exist_ok=True)
    time = np.arange(len(accel), dtype=np.float32) / np.float32(fs)
    np.savez_compressed(ts_dir / f"{event_id}.npz", time=time,
                        accel=accel.astype(np.float32),
                        delta_v=dv_series.astype(np.float32))


def load_synshrp2(data_dir, ts_dir=None):
    "Loads kinematics/<Event_ID>.json + Tabular_records.tab (or a legacy CSV/annotations.csv fallback)."
    log.info("Loading SynSHRP2 ...")
    kin_dir = data_dir / "kinematics"
    if not kin_dir.exists():
        log.warning("SynSHRP2 kinematics dir not found: %s", kin_dir)
        return pd.DataFrame()

    ann_lookup, sev_lookup, impact_lookup = {}, {}, {}
    tab_path = data_dir / "Tabular_records.tab"
    ann_path = data_dir / "annotations.csv"

    if tab_path.exists():
        tab = pd.read_csv(tab_path, sep="\t")
        tab["_key"] = tab["Event_ID"].astype(int).astype(str)
        tab["_event_type"] = tab["Event_type"].map(
            {"Crash": "crash", "Near-Crash": "near_crash"}
        ).fillna("unknown")
        ann_lookup    = tab.set_index("_key")["_event_type"].to_dict()
        sev_lookup    = tab.set_index("_key")["Crash_severity"].fillna("").to_dict()
        impact_lookup = tab.set_index("_key")["Impact"].to_dict()
        log.info("  Annotations: %d events from Tabular_records.tab", len(tab))
    elif ann_path.exists():
        ann = pd.read_csv(ann_path)
        ann_lookup = ann.set_index("event_id")["event_type"].to_dict()
        log.info("  Annotations: %d events from annotations.csv", len(ann))
    else:
        log.warning("  No annotation file found — event_type will be 'unknown'")

    kin_files = sorted(kin_dir.glob("*.json")) + sorted(kin_dir.glob("*.csv"))
    if not kin_files:
        log.warning("No kinematic files in %s", kin_dir)
        return pd.DataFrame()

    target_samples = int(WINDOW_S * TARGET_HZ)
    records = []

    for kin_file in kin_files:
        eid_raw = kin_file.stem
        try:
            if kin_file.suffix == ".json":
                df = pd.DataFrame(json.loads(kin_file.read_text()))
                lon_col, lat_col, ver_col, time_col = "Lon_Acc", "Lat_Acc", "Ver_Acc", "TimeStamp"
            else:
                df = pd.read_csv(kin_file)
                accel_cols = [c for c in df.columns if any(
                    kw in c.lower() for kw in ("accel", "acc_", "acceleration")
                )]
                if len(accel_cols) < 3:
                    nums = df.select_dtypes(include=np.number).columns
                    tl = {c for c in nums if any(
                        kw in c.lower() for kw in ("time", "ts", "timestamp")
                    )}
                    accel_cols = [c for c in nums if c not in tl][:3]
                if len(accel_cols) < 3:
                    log.warning("Skipping %s: <3 accel columns", eid_raw)
                    continue
                lon_col, lat_col, ver_col = accel_cols[:3]
                time_col = next((c for c in df.columns if any(
                    kw in c.lower() for kw in ("time", "ts", "timestamp")
                )), None)

            accel_g = df[[lon_col, lat_col, ver_col]].values.astype(float)

            if time_col and time_col in df.columns:
                t_vals = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(t_vals) & np.all(np.isfinite(accel_g), axis=1)
                t_vals, accel_g = t_vals[valid], accel_g[valid]

                if len(t_vals) < 3:
                    log.warning("Skipping %s: fewer than 3 valid timestamped samples", eid_raw)
                    continue

                ts = pd.DataFrame({"_time": t_vals, "_lon": accel_g[:, 0],
                                   "_lat": accel_g[:, 1], "_ver": accel_g[:, 2]})
                ts = ts.groupby("_time", as_index=False, sort=True).mean(numeric_only=True)
                t_vals = ts["_time"].to_numpy(dtype=float)
                accel_g = ts[["_lon", "_lat", "_ver"]].to_numpy(dtype=float)

                time_diffs = np.diff(t_vals)
                time_diffs = time_diffs[np.isfinite(time_diffs) & (time_diffs > 0)]
                if len(time_diffs) == 0:
                    log.warning("Skipping %s: no increasing timestamps", eid_raw)
                    continue
                dt = float(np.median(time_diffs))

                if kin_file.suffix.lower() == ".json":
                    t_ms, src_hz = t_vals, max(int(round(1000.0 / dt)), 1)
                else:
                    is_ms = "ms" in time_col.lower() or dt >= 10.0
                    if is_ms:
                        t_ms, src_hz = t_vals, max(int(round(1000.0 / dt)), 1)
                    else:
                        t_ms, src_hz = t_vals * 1000.0, max(int(round(1.0 / dt)), 1)

                if src_hz > 1000:
                    log.warning("Skipping %s: implausible sampling rate %d Hz", eid_raw, src_hz)
                    continue
            else:
                src_hz, t_ms = 10, np.arange(len(accel_g)) * 100.0

            impact_ms = impact_lookup.get(eid_raw)
            if pd.notna(impact_ms):
                impact_ms = float(impact_ms)
                mask = (t_ms >= impact_ms - WINDOW_PRE_MS) & (t_ms <= impact_ms + WINDOW_CRASH_MS)
                accel_g = accel_g[mask] if mask.sum() >= 3 else accel_g[-int(WINDOW_S * src_hz):]
            else:
                accel_g = accel_g[-int(WINDOW_S * src_hz):]

            accel_ms2 = resample(accel_g * G_TO_MS2, src_hz)
            accel_ms2 = bandpass(accel_ms2, TARGET_HZ)
            accel_win = pad_or_trim(clip_g(accel_ms2 / G_TO_MS2) * G_TO_MS2, target_samples)

            eid = f"synshrp2_{eid_raw}"
            stats = summarise(accel_win)
            if ts_dir is not None:
                save_timeseries(eid, accel_win, TARGET_HZ, ts_dir, stats.pop("_dv_series"))
            else:
                stats.pop("_dv_series")

            records.append({
                "event_id": eid, "source": "SynSHRP2",
                "event_type": ann_lookup.get(eid_raw, "unknown"),
                "crash_severity": sev_lookup.get(eid_raw, ""),
                **stats,
            })
        except Exception:
            log.warning("Failed: %s\n%s", eid_raw, traceback.format_exc())

    log.info("  Loaded %d SynSHRP2 events", len(records))
    return pd.DataFrame(records)


def load_ciss(data_dir):
    "Loads EDREVENT.csv from NHTSA's bulk CISS export; MAXDVLONG/MAXDVLAT assumed mph, sentinel values (|v|>150) dropped."
    log.info("Loading NHTSA CISS EDR reports ...")

    event_path = data_dir / "EDREVENT.csv"
    if not event_path.exists():
        log.warning("EDREVENT.csv not found in %s", data_dir)
        return pd.DataFrame()

    try:
        events = pd.read_csv(event_path)
    except Exception:
        log.warning("Failed to read EDREVENT.csv\n%s", traceback.format_exc())
        return pd.DataFrame()

    required = {"CASEID", "VEHNO", "EDREVENTNO", "MAXDVLONG", "MAXDVLAT"}
    missing = required - set(events.columns)
    if missing:
        log.warning("EDREVENT.csv missing expected columns: %s", missing)
        return pd.DataFrame()

    def clean_mph(series):
        s = pd.to_numeric(series, errors="coerce")
        return s.where(s.abs() <= PLAUSIBLE_DV_MPH)

    dv_long_kmh = clean_mph(events["MAXDVLONG"]).abs() * MPH_TO_KMH
    dv_lat_kmh  = clean_mph(events["MAXDVLAT"]).abs() * MPH_TO_KMH
    dv_res_kmh  = np.hypot(dv_long_kmh.fillna(0), dv_lat_kmh.fillna(0))
    dv_res_kmh  = dv_res_kmh.where(dv_long_kmh.notna())

    out = pd.DataFrame({
        "event_id": ("ciss_" + events["CASEID"].astype(str)
                     + "_v" + events["VEHNO"].astype(str)
                     + "_e" + events["EDREVENTNO"].astype(str)),
        "source": "CISS", "event_type": "crash",
        "delta_v_longitudinal": dv_long_kmh.round(3),
        "delta_v_lateral"     : dv_lat_kmh.round(3),
        "delta_v_resultant"   : dv_res_kmh.round(3),
        "pre_crash_speed_kmh" : np.nan,
    })
    out["severity_label"] = out["delta_v_resultant"].apply(
        lambda v: classify_severity(v) if pd.notna(v) else "unknown"
    )

    n_total = len(out)
    out = out.dropna(subset=["delta_v_resultant"])
    if n_total - len(out):
        log.info("  Dropped %d/%d events with missing/sentinel delta-V", n_total - len(out), n_total)

    log.info("  Loaded %d CISS EDR events (from %d EDREVENT rows)", len(out), n_total)
    return out


def load_har(data_dir, ts_dir=None):
    "Loads sensoringData_acc.csv (UDC data_raw release), filters to driving rows, windows by wall-clock time per session."
    log.info("Loading Real-Life HAR raw driving segments ...")

    acc_path = data_dir / "sensoringData_acc.csv"
    if not acc_path.exists():
        log.warning("sensoringData_acc.csv not found in %s", data_dir)
        return pd.DataFrame()

    usecols = ["username", "timestamp", "acc_x_axis", "acc_y_axis",
              "acc_z_axis", "activity_id", "activity"]
    driving_chunks = []
    try:
        for chunk in pd.read_csv(acc_path, usecols=usecols, chunksize=HAR_CHUNK_ROWS,
                                 dtype={"username": str, "activity_id": "int64", "activity": str}):
            mask = chunk["activity"].str.contains("driv", case=False, na=False)
            if mask.any():
                driving_chunks.append(chunk.loc[mask])
    except Exception:
        log.warning("Failed to read sensoringData_acc.csv\n%s", traceback.format_exc())
        return pd.DataFrame()

    if not driving_chunks:
        log.warning("No rows with activity containing 'driv' found in %s", acc_path)
        return pd.DataFrame()

    driving = pd.concat(driving_chunks, ignore_index=True)
    del driving_chunks
    log.info("  %d driving-labelled accelerometer samples found", len(driving))

    target_samples = int(WINDOW_S * TARGET_HZ)
    stride_s = WINDOW_S / 2
    records, n_sessions = [], 0

    for (user, session_id), g in driving.groupby(["username", "activity_id"], sort=False):
        n_sessions += 1
        g = g.sort_values("timestamp")
        t_all = g["timestamp"].to_numpy(dtype=float)
        accel_all = g[["acc_x_axis", "acc_y_axis", "acc_z_axis"]].to_numpy(dtype=float)

        finite = np.isfinite(t_all) & np.all(np.isfinite(accel_all), axis=1)
        t_all, accel_all = t_all[finite], accel_all[finite]
        if len(t_all) < 3 or (t_all[-1] - t_all[0]) < WINDOW_S:
            continue

        t1 = t_all[-1]
        start, w_idx = t_all[0], 0
        while start + WINDOW_S <= t1:
            mask = (t_all >= start) & (t_all < start + WINDOW_S)
            if int(mask.sum()) < HAR_MIN_SAMPLES_PER_WINDOW:
                start += stride_s
                continue

            t_win = t_all[mask] - t_all[mask][0]
            accel_win = accel_all[mask]

            dt = np.diff(t_win)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt) == 0:
                start += stride_s
                w_idx += 1
                continue
            src_hz = 1.0 / float(np.median(dt))
            if not (HAR_SRC_HZ_MIN <= src_hz <= HAR_SRC_HZ_MAX):
                start += stride_s
                w_idx += 1
                continue

            try:
                accel_aligned  = align_axes(accel_win)
                accel_resamp   = resample(accel_aligned, src_hz)
                accel_filtered = bandpass(accel_resamp, TARGET_HZ)
                accel_window   = pad_or_trim(
                    clip_g(accel_filtered / G_TO_MS2) * G_TO_MS2, target_samples
                )

                eid = f"har_u{user}_s{session_id}_w{w_idx:04d}"
                stats = summarise(accel_window)
                if ts_dir is not None:
                    save_timeseries(eid, accel_window, TARGET_HZ, ts_dir, stats.pop("_dv_series"))
                else:
                    stats.pop("_dv_series")
                records.append({"event_id": eid, "source": "HAR", **stats})
            except Exception:
                log.warning("Failed: HAR user=%s session=%s window=%d\n%s",
                           user, session_id, w_idx, traceback.format_exc())

            start += stride_s
            w_idx += 1

    log.info("  Loaded %d HAR driving windows from %d driving sessions", len(records), n_sessions)
    return pd.DataFrame(records)


CRASH_COLS = ["event_id", "source", "event_type", "crash_severity",
             "delta_v_longitudinal", "delta_v_lateral", "delta_v_resultant",
             "severity_label", "peak_accel_g", "clipped_ratio",
             "window_duration_s", "sampling_rate_hz"]
NORMAL_COLS = ["event_id", "source",
              "delta_v_longitudinal", "delta_v_lateral", "delta_v_resultant",
              "severity_label", "peak_accel_g", "clipped_ratio",
              "window_duration_s", "sampling_rate_hz"]
VALIDATION_COLS = ["event_id", "edr_delta_v_longitudinal", "edr_delta_v_lateral",
                   "edr_delta_v_resultant", "edr_severity_label", "pre_crash_speed_kmh"]
VALIDATION_RENAME = {
    "delta_v_longitudinal": "edr_delta_v_longitudinal",
    "delta_v_lateral"     : "edr_delta_v_lateral",
    "delta_v_resultant"   : "edr_delta_v_resultant",
    "severity_label"      : "edr_severity_label",
}


def build_datasets(synshrp2_dir, ciss_dir, har_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = output_dir / "timeseries"

    synshrp2_df = load_synshrp2(synshrp2_dir, ts_dir) if synshrp2_dir.exists() else pd.DataFrame()
    if not synshrp2_dir.exists():
        log.warning("SynSHRP2 directory not found: %s", synshrp2_dir)

    ciss_df = load_ciss(ciss_dir) if ciss_dir.exists() else pd.DataFrame()
    if not ciss_dir.exists():
        log.warning("CISS directory not found: %s", ciss_dir)

    har_df = load_har(har_dir, ts_dir) if har_dir.exists() else pd.DataFrame()
    if not har_dir.exists():
        log.warning("HAR directory not found: %s", har_dir)

    crash_frames = [df for df in (synshrp2_df, ciss_df) if not df.empty]
    crashes = (pd.concat(crash_frames, ignore_index=True).reindex(columns=CRASH_COLS)
              if crash_frames else pd.DataFrame(columns=CRASH_COLS))
    crashes.to_csv(output_dir / "crashes.csv", index=False)

    normal = har_df.reindex(columns=NORMAL_COLS) if not har_df.empty else pd.DataFrame(columns=NORMAL_COLS)
    normal.to_csv(output_dir / "normal.csv", index=False)

    if not ciss_df.empty:
        validation = ciss_df.reindex(columns=[
            "event_id", "delta_v_longitudinal", "delta_v_lateral",
            "delta_v_resultant", "severity_label", "pre_crash_speed_kmh",
        ]).rename(columns=VALIDATION_RENAME)
    else:
        validation = pd.DataFrame(columns=VALIDATION_COLS)
    validation.to_csv(output_dir / "validation.csv", index=False)

    log.info("\n=== Output Summary ===")
    log.info("crashes.csv    : %d rows", len(crashes))
    if not crashes.empty:
        log.info("  Sources    : %s", crashes["source"].value_counts().to_dict())
        log.info("  Types      : %s", crashes["event_type"].value_counts().to_dict())
        log.info("  Severity   : %s", crashes["severity_label"].value_counts().to_dict())
        log.info("  Mean clip  : %.1f%%", crashes["clipped_ratio"].dropna().mean() * 100)
    log.info("normal.csv     : %d rows", len(normal))
    log.info("validation.csv : %d rows", len(validation))
    log.info("Files written  : %s", output_dir)

    return crashes, normal, validation


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build crash severity datasets from SynSHRP2, CISS, and HAR.")
    p.add_argument("--synshrp2_dir", type=Path, default=Path("./data/synshrp2"))
    p.add_argument("--ciss_dir",     type=Path, default=Path("./data/ciss"))
    p.add_argument("--har_dir",      type=Path, default=Path("./data/har"))
    p.add_argument("--output_dir",   type=Path, default=Path("./output"))
    args = p.parse_args()
    build_datasets(args.synshrp2_dir, args.ciss_dir, args.har_dir, args.output_dir)
