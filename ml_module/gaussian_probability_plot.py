from pathlib import Path
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, pearsonr, spearmanr



def clean_numeric_series(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values


def clean_values(series):
    values = clean_numeric_series(series)
    values = values.dropna()
    return values.to_numpy(dtype=float)


MIN_SAMPLES = 5
CURVE_POINTS = 500
PSNR_THRESHOLD = 70.0
SSIM_THRESHOLD = 0.99
DWT_BANDS = ["LL", "LH", "HL", "HH"]

OUTPUT_DIR_NAME = "probability_analysis"
FINAL_DATASET_NAME = "probability_intelligence_final_dataset.csv"
SUMMARY_NAME = "probability_intelligence_summary.txt"

def resolve_project_root():
    here = Path(__file__).resolve()
    candidates = [here.parent.parent, here.parent]
    for root in candidates:
        if (root / "ml_module" / "output").exists() or (root / "output").exists():
            return root
    return here.parent.parent

PROJECT_ROOT = resolve_project_root()
PRE_CANDIDATES = [
    PROJECT_ROOT / "ml_module" / "output" / "dataset_builder" / "master_dataset.csv"
]
POST_CANDIDATES = [
    PROJECT_ROOT / "ml_module" / "output" / "post_embedding_analysis" / "actual_post_embedding_final_dataset.csv"
]
DWT_DIR = PROJECT_ROOT / "output" / "dwt_decomposition"
PROFILE_PATH = PROJECT_ROOT / "output" / "image_acquisition" / "image_profile.json"
OUTPUT_DIR = PROJECT_ROOT / "ml_module" / "output" / OUTPUT_DIR_NAME
PLOT_DIR = OUTPUT_DIR / "plots"

def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def load_csv_required(paths, label):
    path = first_existing(paths)
    if path is None:
        searched = "\n".join(str(x) for x in paths)
        raise FileNotFoundError(f"{label} dataset not found. Searched:\n{searched}")
    try:
        return pd.read_csv(path), path
    except Exception as exc:
        raise RuntimeError(f"Could not read {label} dataset:\n{path}\n{exc}") from exc

def load_profile():
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def nested_find(obj, names):
    names = {str(x).lower() for x in names}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in names and isinstance(value, (str, int, float)):
                return value
            result = nested_find(value, names)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = nested_find(value, names)
            if result is not None:
                return result
    return None

def image_shape_from_profile(profile):
    value = nested_find(profile, {"image_path", "original_image_path", "cover_image_path"})
    if value is not None:
        path = Path(str(value))
        if path.exists():
            try:
                from PIL import Image
                with Image.open(path) as image:
                    return image.height, image.width
            except Exception:
                pass
    height = nested_find(profile, {"height", "image_height", "rows"})
    width = nested_find(profile, {"width", "image_width", "columns"})
    try:
        if height and width:
            return int(height), int(width)
    except Exception:
        pass
    return 512, 512

def load_dwt():
    result = {}
    missing = []
    for band in DWT_BANDS:
        path = DWT_DIR / f"{band}.npy"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            array = np.asarray(np.load(path), dtype=float)
            if array.ndim != 2 or array.size == 0:
                raise ValueError("DWT array must be a non-empty 2D array")
            result[band] = array
        except Exception as exc:
            raise RuntimeError(f"Could not load DWT band {band}:\n{path}\n{exc}") from exc
    if not result:
        raise FileNotFoundError(f"No DWT bands were found in {DWT_DIR}")
    return result, missing

def norm_name(value):
    return str(value).strip().lower().replace(" ", "_")

def exact(df, names):
    lookup = {norm_name(c): c for c in df.columns}
    for name in names:
        key = norm_name(name)
        if key in lookup:
            return lookup[key]
    return None

def contains(df, terms, exclude=()):
    for column in df.columns:
        low = norm_name(column)
        if all(term.lower() in low for term in terms) and not any(x.lower() in low for x in exclude):
            return column
    return None

def detect_columns(pre, post):
    d = {}
    d["post_region_id"] = exact(post, ["region_id", "region_index", "region"])
    d["pre_region_id"] = exact(pre, ["region_id", "region_index", "region"])
    d["post_row"] = exact(post, ["pre_row_start", "row_start", "coordinates_y", "region_row"])
    d["post_col"] = exact(post, ["pre_column_start", "column_start", "coordinates_x", "region_column"])
    d["pre_row"] = exact(pre, ["row_start", "pre_row_start", "coordinates_y", "region_row"])
    d["pre_col"] = exact(pre, ["column_start", "pre_column_start", "coordinates_x", "region_column"])
    d["payload"] = exact(post, ["payload_size", "payload", "embedding_payload"])
    d["capacity"] = exact(post, ["capacity", "estimated_capacity", "actual_safe_capacity"])
    d["psnr"] = exact(post, ["image_psnr", "actual_image_psnr", "post_image_psnr"])
    d["ssim"] = exact(post, ["image_ssim", "actual_image_ssim", "post_image_ssim"])
    d["mse"] = exact(post, ["image_mse", "actual_image_mse", "post_image_mse"])
    d["center_distortion"] = exact(post, ["actual_center_distortion", "actual_centre_distortion", "center_distortion", "centre_distortion"])
    d["neighbor_impact"] = exact(post, ["actual_neighbor_impact", "actual_neighbour_impact", "actual_neighbor_mean_impact", "actual_neighbour_mean_impact"])
    d["quality_loss"] = exact(post, ["actual_quality_loss", "quality_loss"])
    d["safe_probability"] = exact(post, ["actual_safe_probability"])
    d["risk_probability"] = exact(post, ["actual_risk_probability"])
    d["band_pre"] = exact(pre, ["band", "subband", "dwt_band", "region_band"])
    d["band_post"] = exact(post, ["band", "subband", "dwt_band", "region_band"])
    return d

def numeric_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def numeric_columns(df):
    result = []
    for c in df.columns:
        s = numeric_series(df[c])
        if s.notna().sum() >= MIN_SAMPLES and s.nunique(dropna=True) > 1:
            result.append(c)
    return result

def canonical_number(value):
    try:
        x = float(value)
        if not np.isfinite(x):
            return None
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.8g}".replace("-", "m").replace(".", "p")
    except Exception:
        return str(value).replace(" ", "_")

def build_spatial_key(df, region_col, row_col, col_col):
    result = pd.Series(index=df.index, dtype="object")
    if row_col and col_col:
        rows = numeric_series(df[row_col])
        cols = numeric_series(df[col_col])
        valid = rows.notna() & cols.notna()
        result.loc[valid] = [
            f"xy_{canonical_number(r)}_{canonical_number(c)}"
            for r, c in zip(rows[valid], cols[valid])
        ]
    if region_col:
        region = df[region_col].astype(str).str.strip()
        missing = result.isna()
        result.loc[missing] = [
            f"rid_{x}" for x in region[missing]
        ]
    missing = result.isna()
    result.loc[missing] = [f"row_{i}" for i in df.index[missing]]
    return result

def safe_stats(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    result = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        f"{prefix}_skew": float(skew(values)) if len(values) >= 3 else 0.0
    }
    return result

def gaussian_probability(values, threshold, higher):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    if std <= 1e-15:
        return float(mean >= threshold) if higher else float(mean <= threshold)
    p = norm.cdf(threshold, mean, std)
    return float(1.0 - p) if higher else float(p)

def empirical_probability(values, threshold, higher):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values >= threshold)) if higher else float(np.mean(values <= threshold))

def fit_probability_features(values, prefix, threshold=None, higher=True, global_values=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {}

    if global_values is None:
        global_values = values

    global_values = np.asarray(global_values, dtype=float)
    global_values = global_values[np.isfinite(global_values)]

    result = safe_stats(values, prefix)

    region_mean = float(np.mean(values))

    if len(global_values) > 1:
        global_mean = float(np.mean(global_values))
        global_std = float(np.std(global_values, ddof=1))

        if global_std > 1e-15:
            result[f"{prefix}_gaussian_pdf"] = float(
                norm.pdf(
                    region_mean,
                    global_mean,
                    global_std
                )
            )

            result[f"{prefix}_gaussian_cdf"] = float(
                norm.cdf(
                    region_mean,
                    global_mean,
                    global_std
                )
            )

            result[f"{prefix}_gaussian_z"] = float(
                (region_mean - global_mean) / global_std
            )
        else:
            result[f"{prefix}_gaussian_pdf"] = 0.0
            result[f"{prefix}_gaussian_cdf"] = 0.5
            result[f"{prefix}_gaussian_z"] = 0.0

        result[f"{prefix}_empirical_cdf"] = float(
            np.mean(
                global_values <= region_mean
            )
        )

    if threshold is not None:
        result[f"{prefix}_gaussian_safety_probability"] = (
            gaussian_probability(
                global_values,
                threshold,
                higher
            )
        )

        result[f"{prefix}_empirical_safety_probability"] = (
            empirical_probability(
                global_values,
                threshold,
                higher
            )
        )

    return result

def distribution_name(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_SAMPLES:
        return "insufficient"
    zero_fraction = float(np.mean(np.isclose(values, 0.0)))
    s = float(abs(skew(values))) if len(values) >= 3 else 0.0
    if zero_fraction >= 0.20:
        return "zero_inflated"
    if s > 1.0:
        return "strongly_skewed"
    if s > 0.5:
        return "moderately_skewed"
    return "approximately_symmetric"

def plot_distribution(values, name, threshold=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_SAMPLES:
        return
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, density=True, alpha=0.45, label="Observed")
    if std > 1e-15:
        lo = min(np.min(values), mean - 4 * std)
        hi = max(np.max(values), mean + 4 * std)
        x = np.linspace(lo, hi, CURVE_POINTS)
        plt.plot(x, norm.pdf(x, mean, std), linewidth=2, label="Gaussian PDF")
    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold={threshold}")
    plt.title(f"{name.upper()} probability density")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}_pdf.png", dpi=180)
    plt.close()
    ordered = np.sort(values)
    empirical = np.arange(1, len(ordered) + 1) / len(ordered)
    plt.figure(figsize=(10, 6))
    plt.plot(ordered, empirical, linewidth=2, label="Empirical CDF")
    if std > 1e-15:
        lo = min(np.min(values), mean - 4 * std)
        hi = max(np.max(values), mean + 4 * std)
        x = np.linspace(lo, hi, CURVE_POINTS)
        plt.plot(x, norm.cdf(x, mean, std), linestyle="--", linewidth=2, label="Gaussian CDF")
    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold={threshold}")
    plt.title(f"{name.upper()} cumulative distribution")
    plt.xlabel(name)
    plt.ylabel("Cumulative probability")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}_cdf.png", dpi=180)
    plt.close()

def aggregate_pre(pre, d):
    work = pre.copy()
    work["_spatial_key"] = build_spatial_key(work, d["pre_region_id"], d["pre_row"], d["pre_col"])
    numeric = numeric_columns(work)
    excluded = {d["pre_region_id"], d["pre_row"], d["pre_col"]}
    numeric = [c for c in numeric if c not in excluded and c != "_spatial_key"]
    band_col = d["band_pre"]
    rows = []
    for key, group in work.groupby("_spatial_key", sort=False, dropna=False):
        record = {"spatial_key": key}
        if d["pre_region_id"]:
            record["region_id"] = str(group[d["pre_region_id"]].iloc[0])
        if d["pre_row"]:
            record["row_start"] = float(numeric_series(group[d["pre_row"]]).dropna().iloc[0]) if numeric_series(group[d["pre_row"]]).notna().any() else np.nan
        if d["pre_col"]:
            record["column_start"] = float(numeric_series(group[d["pre_col"]]).dropna().iloc[0]) if numeric_series(group[d["pre_col"]]).notna().any() else np.nan
        if band_col:
            bands = group[band_col].astype(str).str.upper().str.strip()
            for band in sorted(bands.dropna().unique()):
                sub = group.loc[bands == band]
                for c in numeric:
                    vals = numeric_series(sub[c]).dropna().to_numpy()
                    if len(vals):
                        record[f"pre_{str(band).lower()}_{c}"] = float(np.mean(vals))
        else:
            for c in numeric:
                vals = numeric_series(group[c]).dropna().to_numpy()
                if len(vals):
                    record[f"pre_{c}"] = float(np.mean(vals))
                    if len(vals) > 1 and np.std(vals) > 1e-12:
                        record[f"pre_{c}_within_region_std"] = float(np.std(vals, ddof=1))
        rows.append(record)
    return pd.DataFrame(rows)

def aggregate_post(post, d):
    work = post.copy()
    work["_spatial_key"] = build_spatial_key(work, d["post_region_id"], d["post_row"], d["post_col"])
    metric_map = {
        "psnr": d["psnr"],
        "ssim": d["ssim"],
        "mse": d["mse"],
        "center_distortion": d["center_distortion"],
        "neighbor_impact": d["neighbor_impact"],
        "quality_loss": d["quality_loss"],
        "safe_probability": d["safe_probability"],
        "risk_probability": d["risk_probability"],
    }
    rows = []
    for key, group in work.groupby("_spatial_key", sort=False, dropna=False):
        record = {"spatial_key": key}
        if d["post_region_id"]:
            record["region_id_post"] = str(group[d["post_region_id"]].iloc[0])
        if d["post_row"]:
            vals = numeric_series(group[d["post_row"]]).dropna()
            if len(vals):
                record["row_start_post"] = float(vals.iloc[0])
        if d["post_col"]:
            vals = numeric_series(group[d["post_col"]]).dropna()
            if len(vals):
                record["column_start_post"] = float(vals.iloc[0])
        payload_values = numeric_series(group[d["payload"]]).dropna().to_numpy() if d["payload"] else np.array([])
        if len(payload_values):
            record.update(safe_stats(payload_values, "observed_payload"))
            record["payload_unique_count"] = int(len(np.unique(payload_values)))
            record["payload_min"] = float(np.min(payload_values))
            record["payload_max"] = float(np.max(payload_values))
        if d["capacity"]:
            cap = numeric_series(group[d["capacity"]]).dropna().to_numpy()
            if len(cap):
                record.update(safe_stats(cap, "capacity"))
        for name, column in metric_map.items():
            if not column:
                continue
            vals = numeric_series(group[column]).dropna().to_numpy()
            if not len(vals):
                continue
            record.update(safe_stats(vals, f"post_{name}"))
        for name, column in metric_map.items():
            if not column or not len(payload_values):
                continue
            y = numeric_series(group[column])
            x = numeric_series(group[d["payload"]])
            valid = x.notna() & y.notna()
            if valid.sum() >= 3 and np.std(x[valid]) > 1e-12 and np.std(y[valid]) > 1e-12:
                xv = x[valid].to_numpy()
                yv = y[valid].to_numpy()
                pear = pearsonr(xv, yv)
                spear = spearmanr(xv, yv)
                record[f"payload_{name}_pearson"] = float(pear.statistic)
                record[f"payload_{name}_pearson_p"] = float(pear.pvalue)
                record[f"payload_{name}_spearman"] = float(spear.statistic)
                record[f"payload_{name}_spearman_p"] = float(spear.pvalue)
                slope = np.polyfit(xv, yv, 1)[0]
                record[f"payload_{name}_slope"] = float(slope)
        if d["psnr"] and d["payload"]:
            p = numeric_series(group[d["payload"]])
            y = numeric_series(group[d["psnr"]])
            valid = p.notna() & y.notna()
            if valid.sum() >= 2:
                order = np.argsort(p[valid].to_numpy())
                pv = p[valid].to_numpy()[order]
                yv = y[valid].to_numpy()[order]
                if len(yv) >= 2:
                    diffs = np.diff(yv)
                    record["payload_psnr_degradation_count"] = float(np.sum(diffs < 0))
                    record["payload_psnr_best_observed_payload"] = float(pv[np.argmax(yv)])
        rows.append(record)
    return pd.DataFrame(rows)

def add_global_probability_columns(post_region, d):
    metric_specs = [
        ("psnr", d["psnr"], PSNR_THRESHOLD, True),
        ("ssim", d["ssim"], SSIM_THRESHOLD, True),
        ("mse", d["mse"], None, False),
        ("center_distortion", d["center_distortion"], None, False),
        ("neighbor_impact", d["neighbor_impact"], None, False),
        ("quality_loss", d["quality_loss"], None, False),
    ]
    out = post_region.copy()
    for name, column, threshold, higher in metric_specs:
        base = f"post_{name}_mean"
        if base not in out.columns:
            continue
        global_vals = out[base].dropna().to_numpy(dtype=float)
        if len(global_vals) < MIN_SAMPLES:
            continue
        mean = float(np.mean(global_vals))
        std = float(np.std(global_vals, ddof=1)) if len(global_vals) > 1 else 0.0
        if std > 1e-15:
            out[f"{name}_global_gaussian_percentile"] = out[base].apply(lambda x: float(norm.cdf(x, mean, std)) if pd.notna(x) else np.nan)
        else:
            out[f"{name}_global_gaussian_percentile"] = out[base].apply(lambda x: float(x >= mean) if pd.notna(x) else np.nan)
        ordered = np.sort(global_vals)
        out[f"{name}_global_empirical_percentile"] = out[base].apply(lambda x: float(np.mean(ordered <= x)) if pd.notna(x) else np.nan)

        if threshold is not None:

            if std > 1e-15:

                if higher:
                    gaussian_probability_value = float(
                        1.0 - norm.cdf(
                            threshold,
                            mean,
                            std
                        )
                    )
                else:
                    gaussian_probability_value = float(
                        norm.cdf(
                            threshold,
                            mean,
                            std
                        )
                    )

            else:

                gaussian_probability_value = float(
                    (
                        mean >= threshold
                        if higher
                        else
                        mean <= threshold
                    )
                )

            empirical_probability_value = float(
                (
                    np.mean(
                        global_vals >= threshold
                    )
                    if higher
                    else
                    np.mean(
                        global_vals <= threshold
                    )
                )
            )

            out[
                f"{name}_global_gaussian_safety_probability"
            ] = gaussian_probability_value

            out[
                f"{name}_global_empirical_safety_probability"
            ] = empirical_probability_value
    return out

def calculate_region_probability_curves(post, d):
    work = post.copy()
    work["_spatial_key"] = build_spatial_key(work, d["post_region_id"], d["post_row"], d["post_col"])
    metric_specs = [
        ("psnr", d["psnr"], PSNR_THRESHOLD, True),
        ("ssim", d["ssim"], SSIM_THRESHOLD, True),
        ("mse", d["mse"], None, False),
        ("center_distortion", d["center_distortion"], None, False),
        ("neighbor_impact", d["neighbor_impact"], None, False),
        ("quality_loss", d["quality_loss"], None, False),
    ]
    rows = []
    for key, group in work.groupby("_spatial_key", sort=False, dropna=False):
        record = {"spatial_key": key}
        for name, column, threshold, higher in metric_specs:
            if not column:
                continue
            values = numeric_series(group[column]).dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            global_values = numeric_series(
                work[column]
            ).dropna().to_numpy(dtype=float)

            record.update(
                fit_probability_features(
                    values,
                    f"region_{name}",
                    threshold,
                    higher,
                    global_values
                )
            )
            record[f"region_{name}_distribution"] = distribution_name(values)
        rows.append(record)
    return pd.DataFrame(rows)

def aggregate_neighbor_features(base, d):
    if base.empty:
        return base
    work = base.copy()
    if "row_start" not in work.columns or "column_start" not in work.columns:
        return work
    rows = numeric_series(work["row_start"])
    cols = numeric_series(work["column_start"])
    valid = rows.notna() & cols.notna()
    if valid.sum() < 2:
        return work
    unique_rows = np.sort(rows[valid].unique())
    unique_cols = np.sort(cols[valid].unique())
    row_rank = {float(v): i for i, v in enumerate(unique_rows)}
    col_rank = {float(v): i for i, v in enumerate(unique_cols)}
    lookup = {}
    for idx in work.index:
        if pd.isna(rows.loc[idx]) or pd.isna(cols.loc[idx]):
            continue
        lookup[(row_rank[float(rows.loc[idx])], col_rank[float(cols.loc[idx])])] = idx
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    metric_bases = [c for c in work.columns if c.startswith("post_") and c.endswith("_mean")]
    metric_bases += [c for c in work.columns if c.startswith("region_") and c.endswith("_gaussian_cdf_at_mean")]
    for idx in work.index:
        if pd.isna(rows.loc[idx]) or pd.isna(cols.loc[idx]):
            continue
        key = (row_rank[float(rows.loc[idx])], col_rank[float(cols.loc[idx])])
        neighbors = [lookup[(key[0]+dr,key[1]+dc)] for dr,dc in offsets if (key[0]+dr,key[1]+dc) in lookup]
        work.loc[idx, "neighbor_count"] = len(neighbors)
        if not neighbors:
            continue
        for metric in metric_bases:
            vals = numeric_series(work.loc[neighbors, metric]).dropna().to_numpy()
            current = numeric_series(pd.Series([work.loc[idx, metric]])).iloc[0]
            if len(vals):
                work.loc[idx, f"neighbor_{metric}_mean"] = float(np.mean(vals))
                work.loc[idx, f"neighbor_{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                work.loc[idx, f"neighbor_{metric}_min"] = float(np.min(vals))
                work.loc[idx, f"neighbor_{metric}_max"] = float(np.max(vals))
                if pd.notna(current):
                    work.loc[idx, f"neighbor_{metric}_difference_mean"] = float(np.mean(np.abs(vals-current)))
    return work

def add_dwt_region_features(base, dwt, image_shape):
    if base.empty or "row_start" not in base.columns or "column_start" not in base.columns:
        return base
    h, w = image_shape
    rows = numeric_series(base["row_start"])
    cols = numeric_series(base["column_start"])
    if "region_height" in base.columns:
        heights = numeric_series(base["region_height"])
    else:
        heights = pd.Series(np.nan, index=base.index)
    if "region_width" in base.columns:
        widths = numeric_series(base["region_width"])
    else:
        widths = pd.Series(np.nan, index=base.index)
    valid_rows = np.sort(rows.dropna().unique())
    valid_cols = np.sort(cols.dropna().unique())
    row_step = float(np.median(np.diff(valid_rows))) if len(valid_rows) > 1 and np.any(np.diff(valid_rows) > 0) else None
    col_step = float(np.median(np.diff(valid_cols))) if len(valid_cols) > 1 and np.any(np.diff(valid_cols) > 0) else None
    if row_step is None:
        row_step = h / max(1, len(valid_rows))
    if col_step is None:
        col_step = w / max(1, len(valid_cols))
    for band, array in dwt.items():
        dh, dw = array.shape
        for idx in base.index:
            if pd.isna(rows.loc[idx]) or pd.isna(cols.loc[idx]):
                continue
            rh = float(heights.loc[idx]) if pd.notna(heights.loc[idx]) and heights.loc[idx] > 0 else row_step
            rw = float(widths.loc[idx]) if pd.notna(widths.loc[idx]) and widths.loc[idx] > 0 else col_step
            y0 = int(np.floor(float(rows.loc[idx]) / h * dh))
            y1 = int(np.ceil((float(rows.loc[idx]) + rh) / h * dh))
            x0 = int(np.floor(float(cols.loc[idx]) / w * dw))
            x1 = int(np.ceil((float(cols.loc[idx]) + rw) / w * dw))
            y0 = max(0, min(dh-1, y0))
            x0 = max(0, min(dw-1, x0))
            y1 = max(y0+1, min(dh, y1))
            x1 = max(x0+1, min(dw, x1))
            values = array[y0:y1, x0:x1]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            prefix = f"dwt_{band.lower()}"
            base.loc[idx, f"{prefix}_mean"] = float(np.mean(values))
            base.loc[idx, f"{prefix}_std"] = float(np.std(values))
            base.loc[idx, f"{prefix}_variance"] = float(np.var(values))
            base.loc[idx, f"{prefix}_energy"] = float(np.mean(values**2))
            base.loc[idx, f"{prefix}_abs_mean"] = float(np.mean(np.abs(values)))
            base.loc[idx, f"{prefix}_min"] = float(np.min(values))
            base.loc[idx, f"{prefix}_max"] = float(np.max(values))
            base.loc[idx, f"{prefix}_median"] = float(np.median(values))
            base.loc[idx, f"{prefix}_skew"] = float(skew(values)) if len(values) >= 3 else 0.0
            hist, _ = np.histogram(values, bins=32, density=True)
            p = hist[hist > 0]
            base.loc[idx, f"{prefix}_entropy"] = float(-np.sum((p/p.sum()) * np.log2(p/p.sum()))) if len(p) else 0.0
            if values.size > 1:
                patch = array[y0:y1, x0:x1]
                gy, gx = np.gradient(patch)
                gradient = np.sqrt(gx**2 + gy**2)
                gradient = gradient[np.isfinite(gradient)]
                base.loc[idx, f"{prefix}_gradient_mean"] = float(np.mean(gradient)) if len(gradient) else 0.0
    energy_cols = {band.lower(): f"dwt_{band.lower()}_energy" for band in dwt if f"dwt_{band.lower()}_energy" in base.columns}
    high_cols = [energy_cols[b] for b in ["lh","hl","hh"] if b in energy_cols]
    if high_cols:
        base["dwt_high_frequency_energy"] = base[high_cols].sum(axis=1, min_count=1)
        if "dwt_ll_energy" in base.columns:
            base["dwt_high_frequency_ratio"] = base["dwt_high_frequency_energy"] / (base["dwt_ll_energy"].abs() + base["dwt_high_frequency_energy"].abs() + 1e-12)
    if "dwt_lh_energy" in base.columns and "dwt_hl_energy" in base.columns:
        base["dwt_lh_hl_energy_ratio"] = base["dwt_lh_energy"] / (base["dwt_hl_energy"].abs() + 1e-12)
    if "dwt_lh_energy" in base.columns and "dwt_hh_energy" in base.columns:
        base["dwt_lh_hh_energy_ratio"] = base["dwt_lh_energy"] / (base["dwt_hh_energy"].abs() + 1e-12)
    if "dwt_hl_energy" in base.columns and "dwt_hh_energy" in base.columns:
        base["dwt_hl_hh_energy_ratio"] = base["dwt_hl_energy"] / (base["dwt_hh_energy"].abs() + 1e-12)
    return base

def add_correlation_priors(base, relationships, target_names):
    if relationships.empty:
        return base
    for target in target_names:
        subset = relationships[relationships["post_target"] == target].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("absolute_correlation", ascending=False).head(20)
        score = pd.Series(0.0, index=base.index)
        weight = 0.0
        for _, row in subset.iterrows():
            source = f"pre_{row['pre_feature']}"
            if source not in base.columns:
                source = row["pre_feature"] if row["pre_feature"] in base.columns else None
            if source is None:
                continue
            x = numeric_series(base[source])
            sd = x.std(ddof=1)
            if pd.isna(sd) or sd <= 1e-15:
                continue
            z = (x - x.mean()) / sd
            r = float(row["pearson_correlation"])
            score += z.fillna(0.0) * r
            weight += abs(r)
        if weight > 0:
            base[f"correlation_prior_{target}"] = score / weight
    return base

def clean_final_dataset(df):
    out = df.copy()

    # Convert numeric-looking object columns to numeric.
    for c in out.columns:
        if out[c].dtype == object:
            converted = pd.to_numeric(out[c], errors="coerce")
            if converted.notna().mean() >= 0.95:
                out[c] = converted

    protected = {
        "spatial_key",
        "region_id",
        "region_id_post"
    }

    # Replace infinities, but DO NOT remove columns.
    numeric = [
        c for c in out.columns
        if c not in protected
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    if numeric:
        out[numeric] = out[numeric].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Detect constant columns, but KEEP them.
    constant = [
        c for c in numeric
        if out[c].nunique(dropna=False) <= 1
    ]

    # Detect duplicate columns, but KEEP them.
    seen = {}
    duplicate = []

    for c in numeric:
        key = pd.util.hash_pandas_object(
            out[c],
            index=False
        ).sum()

        if key in seen and out[c].equals(out[seen[key]]):
            duplicate.append(c)
        else:
            seen[key] = c

    # Convert remaining feature-like columns to numeric where possible.
    feature_columns = [
        c for c in out.columns
        if c not in protected
    ]

    for c in feature_columns:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce"
        )

    return out, constant, duplicate

def write_summary(path, pre, post, final, dwt, metrics, relationships):
    groups = {
        "PRE": [c for c in final.columns if c.startswith("pre_")],
        "POST": [c for c in final.columns if c.startswith("post_")],
        "PROBABILITY": [c for c in final.columns if "gaussian" in c.lower() or "empirical" in c.lower() or "percentile" in c.lower()],
        "NEIGHBOUR": [c for c in final.columns if "neighbor" in c.lower() or "neighbour" in c.lower()],
        "DWT": [c for c in final.columns if c.startswith("dwt_")],
        "CORRELATION": [c for c in final.columns if "correlation" in c.lower() or c.startswith("payload_")],
    }
    lines = [
        "StegaQEntropy Probability Intelligence Summary",
        f"PRE source rows: {len(pre)}",
        f"POST source rows: {len(post)}",
        f"Final spatial rows: {len(final)}",
        f"Final columns: {len(final.columns)}",
        f"DWT bands loaded: {', '.join(dwt.keys())}",
        f"PRE→POST correlation relationships: {len(relationships)}",
        "",
        "Detected columns:"
    ]
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    for name, cols in groups.items():
        lines.append(f"{name} features: {len(cols)}")
    lines.append("")
    lines.append("All computed region-level information is stored in the single final CSV.")
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    warnings.filterwarnings("ignore")
    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("StegaQEntropy Probability and Statistical Intelligence")
    print("=" * 75)

    pre, pre_path = load_csv_required(PRE_CANDIDATES, "Pre-embedding")
    post, post_path = load_csv_required(POST_CANDIDATES, "Post-embedding")
    dwt, missing_dwt = load_dwt()
    profile = load_profile()
    image_shape = image_shape_from_profile(profile)
    d = detect_columns(pre, post)

    print(f"\nPre-embedding dataset : {pre.shape}")
    print(f"Post-embedding dataset: {post.shape}")
    print(f"Image shape           : {image_shape}")
    print("\nDetected columns:")
    for k, v in d.items():
        print(f"{k:22s}: {v}")

    print("\nBuilding region-level PRE representation...")
    pre_region = aggregate_pre(pre, d)

    print("Building region-level POST/scenario representation...")
    post_region = aggregate_post(post, d)
    post_region = add_global_probability_columns(post_region, d)

    print("Building region-conditioned probability features...")
    region_probability = calculate_region_probability_curves(post, d)

    print("Calculating PRE→POST correlations...")
    pre_for_corr = pre.copy()
    post_for_corr = post.copy()
    pre_for_corr["_spatial_key"] = build_spatial_key(pre_for_corr, d["pre_region_id"], d["pre_row"], d["pre_col"])
    post_for_corr["_spatial_key"] = build_spatial_key(post_for_corr, d["post_region_id"], d["post_row"], d["post_col"])
    pre_corr_region = aggregate_pre(pre, d)
    post_corr_region = aggregate_post(post, d)
    merged_corr = pre_corr_region.merge(post_corr_region, on="spatial_key", how="inner")
    corr_rows = []
    pre_features = [c for c in pre_corr_region.columns if c.startswith("pre_") and pd.api.types.is_numeric_dtype(pre_corr_region[c])]
    targets = [c for c in post_corr_region.columns if c.startswith("post_") and c.endswith("_mean")]
    for target in targets:
        y = numeric_series(merged_corr[target])
        for feature in pre_features:
            x = numeric_series(merged_corr[feature])
            valid = x.notna() & y.notna()
            if valid.sum() < MIN_SAMPLES:
                continue
            xv = x[valid].to_numpy()
            yv = y[valid].to_numpy()
            if np.std(xv) <= 1e-15 or np.std(yv) <= 1e-15:
                continue
            r, p = pearsonr(xv, yv)
            corr_rows.append({
                "pre_feature": feature,
                "post_target": target,
                "pearson_correlation": float(r),
                "absolute_correlation": abs(float(r)),
                "p_value": float(p),
                "sample_count": int(valid.sum())
            })
    relationships = pd.DataFrame(corr_rows)
    if not relationships.empty:
        relationships = relationships.sort_values("absolute_correlation", ascending=False)

    print("Adding correlation priors...")
    post_targets_for_prior = targets
    pre_region = add_correlation_priors(pre_region, relationships, post_targets_for_prior)

    print("Calculating spatial 8-neighbour features...")

    base = post.copy().reset_index(drop=True)

    base["spatial_key"] = build_spatial_key(
        base,
        d["post_region_id"],
        d["post_row"],
        d["post_col"]
    )

    pre_region_for_merge = pre_region.copy()

    base = base.merge(
        pre_region_for_merge,
        on="spatial_key",
        how="left",
        suffixes=("", "_pre")
    )

    if not region_probability.empty:
        base = base.merge(
            region_probability,
            on="spatial_key",
            how="left",
            suffixes=("", "_prob")
        )
    if len(base) != len(post):
        raise RuntimeError(
            f"ROW EXPANSION ERROR: POST={len(post)}, BASE={len(base)}"
        )

    if d.get("post_row") and d["post_row"] in base.columns:
        base["row_start"] = pd.to_numeric(
            base[d["post_row"]],
            errors="coerce"
        )

    if d.get("post_col") and d["post_col"] in base.columns:
        base["column_start"] = pd.to_numeric(
            base[d["post_col"]],
            errors="coerce"
        )

    base = aggregate_neighbor_features(
        base,
        d
    )

    print("Extracting region-level DWT features...")

    base = add_dwt_region_features(
        base,
        dwt,
        image_shape
    )

    if "region_id_post" in base.columns and "region_id" not in base.columns:
        base["region_id"] = base["region_id_post"]

    post_target_columns = [
        c for c in base.columns
        if c.startswith("post_")
    ]

    base = base.rename(
        columns={
            c: f"target_{c}"
            for c in post_target_columns
        }
    )

    final, constant_columns, duplicate_columns = clean_final_dataset(base)

    if len(final) != len(post):
        raise RuntimeError(
            f"FINAL ROW COUNT ERROR: POST={len(post)}, FINAL={len(final)}"
        )

    protected = [
        c for c in [
            "spatial_key",
            "region_id",
            "region_id_post",
            "payload_size",
            "capacity",
            "image_psnr",
            "image_ssim",
            "image_mse",
            "actual_center_distortion",
            "actual_neighbor_impact",
            "actual_quality_loss",
            "actual_safe_probability",
            "actual_risk_probability"
        ]
        if c in final.columns
    ]

    target_columns = [
        c for c in final.columns
        if c.startswith("target_post_")
    ]

    feature_columns = [
        c for c in final.columns
        if c not in protected
        and c not in target_columns
    ]

    if not feature_columns:
        raise RuntimeError("No usable numeric features were produced.")

    final_path = OUTPUT_DIR / FINAL_DATASET_NAME
    final.to_csv(final_path, index=False)

    summary_path = OUTPUT_DIR / SUMMARY_NAME
    write_summary(summary_path, pre, post, final, dwt, d, relationships)

    for name, column, threshold in [
        ("psnr", d["psnr"], PSNR_THRESHOLD),
        ("ssim", d["ssim"], SSIM_THRESHOLD),
        ("mse", d["mse"], None),
        ("center_distortion", d["center_distortion"], None),
        ("neighbor_impact", d["neighbor_impact"], None),
        ("quality_loss", d["quality_loss"], None)
    ]:
        if column:
            values = clean_numeric_series(post[column]).dropna().to_numpy()
            if len(values) >= MIN_SAMPLES:
                plot_distribution(values, name, threshold)

    print("\n" + "=" * 75)
    print("FINAL PROBABILITY INTELLIGENCE DATASET")
    print("=" * 75)
    print(f"PRE source rows       : {len(pre)}")
    print(f"POST source rows      : {len(post)}")
    print(f"Final spatial rows    : {len(final)}")
    print(f"Final total columns   : {len(final.columns)}")
    print(f"Final ML features     : {len(feature_columns)}")
    print(f"Final target columns  : {len(target_columns)}")
    print(f"DWT bands             : {', '.join(dwt.keys())}")
    print(f"DWT missing           : {len(missing_dwt)}")
    print(f"PRE→POST correlations : {len(relationships)}")
    print(f"Detected constants    : {len(constant_columns)}")
    print(f"Detected duplicates   : {len(duplicate_columns)}")
    print(f"\nFINAL CSV:\n{final_path}")
    print(f"\nSUMMARY:\n{summary_path}")
    print(f"\nPLOTS:\n{PLOT_DIR}")
    print("\nAll computed region-level information is consolidated into the final CSV.")

if __name__ == "__main__":
    main()
