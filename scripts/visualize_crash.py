"""Renders one event as a tabbed Plotly HTML: Force Trajectory, Vehicle Path, Delta-V Timeline, Similar Events."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

SEV_COLOUR = {"low": "#1baf7a", "moderate": "#eda100", "severe": "#e24b4a"}
SEV_THRESHOLDS = {"low": 15.0, "moderate": 35.0}


def load_event(output_dir, event_id):
    """CISS events have no .npz (scalar delta-V only) so they never reach this."""
    ts_path = output_dir / "timeseries" / f"{event_id}.npz"
    if not ts_path.exists():
        sys.exit(f"No time-series file found for event '{event_id}'.\n"
                 f"Expected: {ts_path}\n"
                 f"Run build_crash_dataset.py first, or use --list to see available events.")
    arrays = np.load(ts_path)

    meta = None
    for csv_name in ("crashes.csv", "normal.csv"):
        csv_path = output_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row = df[df["event_id"] == event_id]
            if not row.empty:
                meta = row.iloc[0].to_dict()
                break
    if meta is None:
        sys.exit(f"Event '{event_id}' not found in crashes.csv or normal.csv.")

    return {"meta": meta, "time": arrays["time"],
           "accel": arrays["accel"], "delta_v": arrays["delta_v"]}


def compute_trajectory(accel_ms2, time_s):
    """Dead-reckoning position via double integration; drifts over the window, no GPS correction."""
    velocity = np.zeros_like(accel_ms2)
    position = np.zeros_like(accel_ms2)
    for axis in range(3):
        velocity[:, axis] = cumulative_trapezoid(accel_ms2[:, axis], time_s, initial=0.0)
        position[:, axis] = cumulative_trapezoid(velocity[:, axis], time_s, initial=0.0)
    return position, velocity


def find_similar_crashes(output_dir, dv_resultant, current_id, n=5):
    frames = []
    crashes_path = output_dir / "crashes.csv"
    val_path = output_dir / "validation.csv"

    if crashes_path.exists():
        df = pd.read_csv(crashes_path)
        df = df[df["event_id"] != current_id]
        frames.append(df[["event_id", "source", "event_type", "delta_v_resultant", "severity_label"]])

    if val_path.exists():
        vdf = pd.read_csv(val_path)
        vdf = vdf.rename(columns={"edr_delta_v_resultant": "delta_v_resultant",
                                  "edr_severity_label": "severity_label"})
        vdf["source"], vdf["event_type"] = "CISS (EDR)", "crash"
        frames.append(vdf[["event_id", "source", "event_type", "delta_v_resultant", "severity_label"]])

    if not frames:
        return pd.DataFrame()

    all_events = pd.concat(frames, ignore_index=True)
    # an empty validation.csv reads back as object dtype, which upcasts the merged column and breaks nsmallest()
    all_events["delta_v_resultant"] = pd.to_numeric(all_events["delta_v_resultant"], errors="coerce")
    all_events = all_events.dropna(subset=["delta_v_resultant"])
    if all_events.empty:
        return pd.DataFrame()

    all_events["distance"] = (all_events["delta_v_resultant"] - dv_resultant).abs()
    return all_events.nsmallest(n, "distance").reset_index(drop=True)


def plot_event(output_dir, event_id):
    import plotly.graph_objects as go

    event     = load_event(output_dir, event_id)
    meta      = event["meta"]
    time      = event["time"]
    accel_g   = event["accel"] / 9.80665
    dv        = event["delta_v"]
    sev       = meta.get("severity_label", "unknown")
    dv_result = meta.get("delta_v_resultant", float(np.max(np.abs(dv))))
    sev_hex   = SEV_COLOUR.get(sev, "#888780")
    similar   = find_similar_crashes(output_dir, dv_result, event_id)

    x, y, z  = accel_g[:, 0], accel_g[:, 1], accel_g[:, 2]
    peak_idx = int(np.argmax(np.abs(x)))

    def rgba(hexcolor, alpha):
        return f"rgba({int(hexcolor[1:3],16)},{int(hexcolor[3:5],16)},{int(hexcolor[5:7],16)},{alpha})"

    PANEL_LAYOUT = dict(
        paper_bgcolor="#f8f8f6", plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(bgcolor="rgba(248,248,246,0.85)", bordercolor="#e1e0d9",
                    borderwidth=1, font=dict(size=10)),
    )

    # Force Trajectory
    hover_accel = [
        f"t = {t:.3f} s<br>Long: {xi:.3f} g<br>Lat: {yi:.3f} g<br>Vert: {zi:.3f} g<br>Δv: {d:.2f} km/h"
        for t, xi, yi, zi, d in zip(time, x, y, z, dv)
    ]
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=3, color=time, colorscale="Plasma",
                    colorbar=dict(title=dict(text="Time (s)", font=dict(size=11)),
                                  thickness=14, len=0.6, tickfont=dict(size=9)),
                    opacity=0.85),
        text=hover_accel, hoverinfo="text", name="Trajectory",
    ))
    fig_accel.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color="rgba(150,150,150,0.35)", width=2),
        hoverinfo="skip", showlegend=False,
    ))
    fig_accel.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=7, color="#888780", symbol="circle"),
        name="Origin (rest)", hovertemplate="Origin (0, 0, 0) g<extra></extra>",
    ))
    fig_accel.add_trace(go.Scatter3d(
        x=[x[peak_idx]], y=[y[peak_idx]], z=[z[peak_idx]], mode="markers",
        marker=dict(size=10, color=sev_hex, symbol="diamond", line=dict(color="white", width=1)),
        name=f"Peak accel ({x[peak_idx]:.2f}, {y[peak_idx]:.2f}, {z[peak_idx]:.2f}) g",
        hovertemplate=(f"Peak sample<br>Long: {x[peak_idx]:.3f} g<br>"
                       f"Lat: {y[peak_idx]:.3f} g<br>Vert: {z[peak_idx]:.3f} g<br>"
                       f"t = {time[peak_idx]:.3f} s<extra></extra>"),
    ))
    fig_accel.add_trace(go.Cone(
        x=[x[peak_idx] * 0.85], y=[y[peak_idx] * 0.85], z=[z[peak_idx] * 0.85],
        u=[x[peak_idx] * 0.15], v=[y[peak_idx] * 0.15], w=[z[peak_idx] * 0.15],
        colorscale=[[0, sev_hex], [1, sev_hex]], showscale=False, sizemode="absolute",
        sizeref=max(np.abs([x[peak_idx], y[peak_idx], z[peak_idx]])) * 0.25,
        opacity=0.7, hovertemplate="PDOF vector<extra></extra>", name="PDOF",
    ))
    fig_accel.add_trace(go.Scatter3d(
        x=[0, x[peak_idx] * 0.85], y=[0, y[peak_idx] * 0.85], z=[0, z[peak_idx] * 0.85],
        mode="lines", line=dict(color=sev_hex, width=4), opacity=0.6,
        hoverinfo="skip", showlegend=False,
    ))
    fig_accel.update_layout(
        **PANEL_LAYOUT, height=650,
        scene=dict(
            xaxis=dict(title="Longitudinal (g)", showgrid=True, gridcolor="#e1e0d9",
                       zeroline=True, zerolinecolor="#c3c2b7"),
            yaxis=dict(title="Lateral (g)", showgrid=True, gridcolor="#e1e0d9",
                       zeroline=True, zerolinecolor="#c3c2b7"),
            zaxis=dict(title="Vertical (g)", showgrid=True, gridcolor="#e1e0d9",
                       zeroline=True, zerolinecolor="#c3c2b7"),
            bgcolor="#fafafa", camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)), aspectmode="cube",
        ),
    )

    # Vehicle Path
    position_m, velocity_ms = compute_trajectory(event["accel"], time)
    speed_kmh = np.linalg.norm(velocity_ms, axis=1) * 3.6
    px, py, pz = position_m[:, 0], position_m[:, 1], position_m[:, 2]
    traj_hover = [
        f"t = {t:.3f} s<br>Position: ({xi:.1f}, {yi:.1f}, {zi:.1f}) m<br>Est. speed: {s:.1f} km/h"
        for t, xi, yi, zi, s in zip(time, px, py, pz, speed_kmh)
    ]
    fig_path = go.Figure()
    fig_path.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, mode="markers",
        marker=dict(size=3, color=time, colorscale="Plasma", showscale=False, opacity=0.85),
        text=traj_hover, hoverinfo="text", name="Estimated path",
    ))
    fig_path.add_trace(go.Scatter3d(
        x=px, y=py, z=pz, mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=2),
        hoverinfo="skip", showlegend=False,
    ))
    fig_path.add_trace(go.Scatter3d(
        x=[px[0]], y=[py[0]], z=[pz[0]], mode="markers",
        marker=dict(size=8, color="#1baf7a", symbol="circle", line=dict(color="white", width=1)),
        name="Start of window", hovertemplate="Start (t=0)<extra></extra>",
    ))
    fig_path.add_trace(go.Scatter3d(
        x=[px[peak_idx]], y=[py[peak_idx]], z=[pz[peak_idx]], mode="markers",
        marker=dict(size=10, color=sev_hex, symbol="diamond", line=dict(color="white", width=1)),
        name="Impact point", hovertemplate=f"Impact — t = {time[peak_idx]:.3f} s<extra></extra>",
    ))
    fig_path.add_trace(go.Scatter3d(
        x=[px[-1]], y=[py[-1]], z=[pz[-1]], mode="markers",
        marker=dict(size=8, color="#0b0b0b", symbol="square"),
        name="End of window", hovertemplate="End of window<extra></extra>",
    ))
    fig_path.update_layout(
        **PANEL_LAYOUT, height=650,
        scene=dict(
            xaxis=dict(title="Longitudinal position (m)", showgrid=True, gridcolor="#e1e0d9",
                       zeroline=True, zerolinecolor="#c3c2b7"),
            yaxis=dict(title="Lateral position (m)", showgrid=True, gridcolor="#e1e0d9",
                       zeroline=True, zerolinecolor="#c3c2b7"),
            zaxis=dict(title="Vertical position (m)", showgrid=True, gridcolor="#e1e0d9",
                       zeroline=True, zerolinecolor="#c3c2b7"),
            bgcolor="#fafafa", camera=dict(eye=dict(x=1.3, y=-2.2, z=1.0)), aspectmode="data",
        ),
    )

    # Delta-V Timeline
    fig_dv = go.Figure()
    fig_dv.add_trace(go.Scatter(
        x=time, y=dv, mode="lines", line=dict(color=sev_hex, width=2.5),
        fill="tozeroy", fillcolor=rgba(sev_hex, 0.12), name="Δv longitudinal",
        hovertemplate="t = %{x:.3f} s<br>Δv = %{y:.2f} km/h<extra></extra>",
    ))
    threshold_colours = {"low": "#1baf7a", "moderate": "#eda100"}
    for label, thr in SEV_THRESHOLDS.items():
        fig_dv.add_hline(
            y=thr, line=dict(color=threshold_colours[label], width=1.2, dash="dash"),
            annotation_text=f"{label} threshold ({thr} km/h)",
            annotation_font=dict(size=9, color=threshold_colours[label]),
            annotation_position="top left",
        )
    fig_dv.update_layout(
        **PANEL_LAYOUT, height=480,
        xaxis=dict(title="Time (s)", gridcolor="#e1e0d9", showgrid=True),
        yaxis=dict(title="Δv (km/h)", gridcolor="#e1e0d9", showgrid=True),
    )

    # Similar Events
    fig_similar = go.Figure()
    if not similar.empty:
        bar_labels  = [f"{r['event_id'][:24]} ({r['source']})" for _, r in similar.iterrows()]
        bar_vals    = similar["delta_v_resultant"].tolist()
        bar_colours = [SEV_COLOUR.get(s, "#888780") for s in similar["severity_label"]]
        fig_similar.add_trace(go.Bar(
            x=bar_vals, y=bar_labels, orientation="h",
            marker=dict(color=bar_colours, opacity=0.85),
            text=[f"{v:.1f}" for v in bar_vals], textposition="outside", textfont=dict(size=10),
            hovertemplate="%{y}<br>Δv = %{x:.1f} km/h<extra></extra>", name="Similar events",
        ))
        fig_similar.add_vline(
            x=dv_result, line=dict(color="#0b0b0b", width=1.5, dash="dash"),
            annotation_text=f"This event: {dv_result:.1f} km/h",
            annotation_font=dict(size=10), annotation_position="top",
        )
    else:
        fig_similar.add_annotation(
            text="No comparison events available yet.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#898781"),
        )
    fig_similar.update_layout(
        **PANEL_LAYOUT, height=480,
        xaxis=dict(title="Δv resultant (km/h)", gridcolor="#e1e0d9", showgrid=True),
        yaxis=dict(tickfont=dict(size=10)),
    )

    # Assemble tabbed HTML
    figures = {"dv": fig_dv, "accel": fig_accel, "path": fig_path, "similar": fig_similar}
    tab_labels = {"dv": "Delta-V Timeline", "accel": "Force Trajectory",
                  "path": "Vehicle Path", "similar": "Similar Events"}
    fig_json = {key: fig.to_json() for key, fig in figures.items()}

    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if k == "dv" else ""}" '
        f'id="btn-{k}" onclick="showTab(\'{k}\')">{label}</button>'
        for k, label in tab_labels.items()
    )
    tab_panels = "\n".join(
        f'<div class="tab-content{" active" if k == "dv" else ""}" id="tab-{k}">'
        f'<div class="plot-container" id="plot-{k}"></div>'
        + ('<div class="caveat">Dead-reckoning estimate from acceleration only — '
           'no GPS/wheel-speed correction. Shape is informative; absolute distance '
           'drifts over the window.</div>' if k == "path" else "")
        + '</div>'
        for k in tab_labels
    )
    fig_json_js = "\n".join(f"const fig_{k} = {v};" for k, v in fig_json.items())

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Crash Event: {event_id}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; background: #f8f8f6; margin: 0; padding: 0; color: #0b0b0b; }}
  .header {{ padding: 16px 24px; border-bottom: 1px solid #e1e0d9; }}
  .header h1 {{ font-size: 17px; margin: 0 0 4px; font-weight: bold; }}
  .header .dot {{ color: {sev_hex}; }}
  .header .meta {{ font-size: 13px; color: #52514e; }}
  .tabs {{ display: flex; gap: 4px; padding: 0 24px; border-bottom: 1px solid #e1e0d9; background: #f8f8f6; }}
  .tab-btn {{ padding: 10px 18px; border: none; background: transparent; cursor: pointer;
              font-size: 13px; color: #52514e; border-bottom: 3px solid transparent; font-family: inherit; }}
  .tab-btn:hover {{ color: #0b0b0b; }}
  .tab-btn.active {{ color: #0b0b0b; font-weight: bold; border-bottom-color: {sev_hex}; }}
  .tab-content {{ display: none; padding: 16px 24px; }}
  .tab-content.active {{ display: block; }}
  .plot-container {{ width: 100%; }}
  .caveat {{ font-size: 11px; color: #898781; margin-top: 8px; }}
</style>
</head>
<body>
  <div class="header">
    <h1>Crash Event: {event_id} &nbsp; <span class="dot">●</span> {sev.upper()}</h1>
    <div class="meta">Δv = {dv_result:.1f} km/h &nbsp;|&nbsp; source: {meta.get('source', '—')}
      &nbsp;|&nbsp; type: {meta.get('event_type', '—')}</div>
  </div>
  <div class="tabs">
{tab_buttons}
  </div>
{tab_panels}
  <script>
{fig_json_js}
    const figures = {{ accel: fig_accel, path: fig_path, dv: fig_dv, similar: fig_similar }};
    const rendered = {{}};
    function showTab(name) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      document.getElementById('btn-' + name).classList.add('active');
      const divId = 'plot-' + name;
      if (!rendered[name]) {{
        Plotly.newPlot(divId, figures[name].data, figures[name].layout, {{responsive: true}});
        rendered[name] = true;
      }} else {{
        Plotly.Plots.resize(document.getElementById(divId));
      }}
    }}
    showTab('dv');
  </script>
</body>
</html>"""

    out_path = output_dir / f"plot_{event_id}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")
    print("Open the .html file in a browser — use the tabs to switch views.")


def list_events(output_dir):
    for csv_name in ("crashes.csv", "normal.csv"):
        p = output_dir / csv_name
        if p.exists():
            df = pd.read_csv(p)
            print(f"\n{csv_name} ({len(df)} events):")
            print(df[["event_id", "source", "event_type",
                       "delta_v_resultant", "severity_label"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a crash event from the dataset.")
    parser.add_argument("--output_dir", type=Path, default=Path("./output"))
    parser.add_argument("--event_id", type=str, default=None)
    parser.add_argument("--random", action="store_true", help="pick a random event")
    parser.add_argument("--include_normal", action="store_true", help="with --random, allow normal.csv too")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_events(args.output_dir)
        sys.exit(0)

    if args.event_id is None:
        ts_dir = args.output_dir / "timeseries"
        available = {f.stem for f in ts_dir.glob("*.npz")} if ts_dir.exists() else set()
        if not available:
            sys.exit("No time-series files found. Run build_crash_dataset.py first.")

        frames = []
        crashes_path = args.output_dir / "crashes.csv"
        if crashes_path.exists():
            frames.append(pd.read_csv(crashes_path))
        if args.random and args.include_normal:
            normal_path = args.output_dir / "normal.csv"
            if normal_path.exists():
                frames.append(pd.read_csv(normal_path))

        if not frames:
            sys.exit("crashes.csv not found. Run build_crash_dataset.py first.")

        df = pd.concat(frames, ignore_index=True)
        df = df[df["event_id"].isin(available)]
        if df.empty:
            sys.exit("No time-series files found for any listed event. Re-run build_crash_dataset.py.")

        if args.random:
            picked = df.sample(n=1, random_state=args.seed).iloc[0]
            args.event_id = picked["event_id"]
            dv = picked.get("delta_v_resultant", float("nan"))
            print(f"--random: picked {args.event_id} (source={picked.get('source', '—')}, Δv={dv:.1f} km/h)")
        else:
            args.event_id = df.loc[df["delta_v_resultant"].idxmax(), "event_id"]
            print(f"No event specified — plotting most severe: {args.event_id}")

    plot_event(args.output_dir, args.event_id)
