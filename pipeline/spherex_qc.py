"""SPHEREx Level-2 QC primitives.

This module is deliberately split into:
  * dependency-free logic (flags, band mapping, ADQL builders, PM, scoring);
  * optional helpers that import numpy/astropy only when called.

Run the self-test without installing astronomy packages:

    python pipeline/spherex_qc.py selftest

It is a scaffold, not a complete pipeline. The end-to-end flow and the FITS
reads are documented in docs/spherex-pipeline.md.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# FLAGS: current public SPHEREx bit assignments (read MSKN*/MSKD* at runtime!)
# ---------------------------------------------------------------------------

FLAG_BITS: dict[str, int] = {
    "TRANSIENT": 0,
    "OVERFLOW": 1,
    "SUR_ERROR": 2,
    "NONFUNC": 6,
    "DICHROIC": 7,
    "MISSING_DATA": 9,
    "HOT": 10,
    "COLD": 11,
    "FULLSAMPLE": 12,
    "PHANMISS": 14,
    "NONLINEAR": 15,
    "PERSIST": 17,
    "OUTLIER": 19,
    "CROSSTALK": 20,
    "SOURCE": 21,
    "GHOST": 22,
    "GHOST_FPA": 23,
    "GHOST_EXT": 24,
    "STREAK": 25,
    "BLOOM": 26,
    "SNOWBALL": 27,
    "HALO": 28,
    "SATELLITE_HALO": 29,
    "CONTAINS_BAD_PIXEL": 32,
    "FIT_ERROR": 33,
}

FLAG_DESCRIPTIONS: dict[str, str] = {
    "TRANSIENT": "transient/cosmic ray found by SUR",
    "OVERFLOW": "pixel near saturation",
    "SUR_ERROR": "on-board processing error",
    "NONFUNC": "permanently non-functional pixel",
    "DICHROIC": "low efficiency due to dichroic",
    "MISSING_DATA": "on-board data lost",
    "HOT": "hot/noisy pixel",
    "COLD": "anomalously low pixel",
    "FULLSAMPLE": "full sample history available (informational)",
    "PHANMISS": "phantom correction not applied",
    "NONLINEAR": "non-linearity correction unreliable",
    "PERSIST": "persistent charge above threshold",
    "OUTLIER": "outlier detected",
    "CROSSTALK": "crosstalk",
    "SOURCE": "pixel matched to a known source (not a defect)",
    "GHOST": "ghost from a bright source in the exposure",
    "GHOST_FPA": "ghost on another detector (usually not in L2 images)",
    "GHOST_EXT": "ghost from a source outside the field",
    "STREAK": "streak/track",
    "BLOOM": "blooming from saturation",
    "SNOWBALL": "snowball / particle glint",
    "HALO": "halo around a transient",
    "SATELLITE_HALO": "satellite/streak halo",
    "CONTAINS_BAD_PIXEL": "aperture contained a bad pixel (derived)",
    "FIT_ERROR": "photometric fit failed to converge (derived)",
}

# Groups used by the UI and by targeted masks.
MASK_GROUPS: dict[str, tuple[str, ...]] = {
    "cosmic": ("TRANSIENT", "SUR_ERROR", "OUTLIER", "HALO", "SATELLITE_HALO"),
    "saturation": ("OVERFLOW", "NONLINEAR", "BLOOM"),
    "bad_pixel": ("NONFUNC", "DICHROIC", "MISSING_DATA", "HOT", "COLD", "PHANMISS", "PERSIST"),
    "artifact": ("CROSSTALK", "GHOST", "GHOST_FPA", "GHOST_EXT", "STREAK", "SNOWBALL"),
    "source": ("SOURCE",),
}

INFORMATIONAL_BITS: tuple[str, ...] = ("FULLSAMPLE",)
DERIVED_BITS: tuple[str, ...] = ("CONTAINS_BAD_PIXEL", "FIT_ERROR")


def bit(name: str) -> int:
    """Return the 1<<n mask for a known flag name."""
    try:
        return 1 << FLAG_BITS[name]
    except KeyError as exc:
        raise KeyError(f"unknown SPHEREx flag: {name!r}") from exc


def bits_for(names: Iterable[str]) -> int:
    total = 0
    for name in names:
        total |= bit(name)
    return total


BAD_FOR_DETECTION = bits_for(
    MASK_GROUPS["cosmic"]
    + MASK_GROUPS["saturation"]
    + MASK_GROUPS["bad_pixel"]
    + MASK_GROUPS["artifact"]
)
BAD_FOR_PHOTOMETRY = BAD_FOR_DETECTION | bits_for(DERIVED_BITS)


def decode_flags(value: int, include_unknown: bool = False) -> list[str]:
    """Decode a flag value into names; combined bits are preserved.

    Normalises negative signed values so that high bits (32/33) survive.
    """
    value = int(value)
    if value < 0:
        value &= 0xFFFFFFFFFFFFFFFF
    names = [name for name, n in sorted(FLAG_BITS.items(), key=lambda kv: kv[1])
             if value & (1 << n)]
    if include_unknown:
        known = set(FLAG_BITS.values())
        for n in range(0, 64):
            if value & (1 << n) and n not in known:
                names.append(f"UNKNOWN_{n}")
    return names


def flags_from_header(header: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    """Read MSKNnnnn/MSKDnnnn from a FITS header-like mapping.

    This is the authoritative flag definition for the file in hand. Use it
    instead of trusting a hard-coded table after a release change.
    """
    out: dict[int, dict[str, str]] = {}
    for key in header:
        text = str(key)
        if text.startswith("MSKN") and len(text) >= 8:
            suffix = text[4:]
            if not suffix.isdigit():
                continue
            n = int(suffix)
            out[n] = {
                "name": str(header[key]).strip(),
                "description": str(header.get("MSKD" + suffix, "")).strip(),
            }
    return dict(sorted(out.items()))


def build_bad_mask(flags: Any, bits: int = BAD_FOR_DETECTION, dilate: int = 0) -> Any:
    """Boolean bad-pixel mask from a FLAGS array.

    Requires numpy. `dilate` is a radius in pixels (0 = no dilation).
    """
    import numpy as np

    arr = np.asarray(flags).astype(np.int64, copy=False)
    mask = (arr & np.int64(bits)) != 0
    if dilate <= 0:
        return mask
    return dilate_mask(mask, radius=int(dilate))


def dilate_mask(mask: Any, radius: int = 1) -> Any:
    """Square/circular dilation without scipy; radius is in pixels."""
    import numpy as np

    arr = np.asarray(mask).astype(bool)
    if radius <= 0:
        return arr.copy()
    h, w = arr.shape
    padded = np.pad(arr, radius, mode="constant", constant_values=False)
    out = np.zeros_like(arr)
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
                continue
            out |= padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return out


# ---------------------------------------------------------------------------
# Bands, detectors and the 102-channel geometry
# ---------------------------------------------------------------------------

BANDS: dict[int, dict[str, Any]] = {
    1: {"name": "D1", "lambda_um": (0.75, 1.09), "R": 39, "channels": 17},
    2: {"name": "D2", "lambda_um": (1.10, 1.62), "R": 41, "channels": 17},
    3: {"name": "D3", "lambda_um": (1.63, 2.41), "R": 41, "channels": 17},
    4: {"name": "D4", "lambda_um": (2.42, 3.82), "R": 35, "channels": 17},
    5: {"name": "D5", "lambda_um": (3.83, 4.41), "R": 112, "channels": 17},
    6: {"name": "D6", "lambda_um": (4.42, 5.00), "R": 128, "channels": 17},
}


def band_index(detector: int, channel: int) -> int:
    """(D1,ch1) -> 1 ... (D6,ch17) -> 102. Geometry only, not calibration."""
    if not 1 <= int(detector) <= 6:
        raise ValueError("detector must be 1..6")
    if not 1 <= int(channel) <= 17:
        raise ValueError("channel must be 1..17; 0/18 mean outside science channels")
    return (int(detector) - 1) * 17 + int(channel)


def detector_channel(index: int) -> tuple[int, int]:
    """Inverse of band_index."""
    if not 1 <= int(index) <= 102:
        raise ValueError("band index must be 1..102")
    return (int(index) - 1) // 17 + 1, (int(index) - 1) % 17 + 1


def band_metadata(detector: int) -> dict[str, Any]:
    if int(detector) not in BANDS:
        raise ValueError("detector must be 1..6")
    return dict(BANDS[int(detector)])


def approx_lambda(detector: int, channel: int) -> float:
    """Linear approximation only. Science must use CWAVE/CBAND."""
    lo, hi = BANDS[int(detector)]["lambda_um"]
    return lo + (hi - lo) * (int(channel) - 0.5) / 17.0


# ---------------------------------------------------------------------------
# ADQL builders (numeric inputs are validated; no user SQL fragments)
# ---------------------------------------------------------------------------

def _num(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_sky(ra: float, dec: float, radius_deg: float) -> tuple[float, float, float]:
    ra, dec, radius_deg = _num(ra, "ra"), _num(dec, "dec"), _num(radius_deg, "radius_deg")
    if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0 or radius_deg <= 0:
        raise ValueError("invalid sky coordinates or radius")
    return ra, dec, radius_deg


def adql_allwise(ra: float, dec: float, radius_deg: float, limit: int = 1000) -> str:
    ra, dec, radius_deg = _validate_sky(ra, dec, radius_deg)
    return f"""
SELECT TOP {int(limit)} designation, ra, dec, sigra, sigdec,
       w1mpro, w1sigmpro, w2mpro, w2sigmpro, w3mpro, w4mpro,
       ph_qual, cc_flags, ext_flg
FROM allwise_p3as_psd
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
""".strip()


def adql_catwise(ra: float, dec: float, radius_deg: float, limit: int = 1000) -> str:
    ra, dec, radius_deg = _validate_sky(ra, dec, radius_deg)
    return f"""
SELECT TOP {int(limit)} source_name, ra, dec, sigra, sigdec,
       w1mpro, w1sigmpro, w2mpro, w2sigmpro,
       pmra, pmdec, sigpmra, sigpmdec, snr1, snr2, ph_qual
FROM catwise_2020
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
  AND snr1 > 5
""".strip()


def adql_gaia(ra: float, dec: float, radius_deg: float, limit: int = 1000,
              g_max: float = 20.5) -> str:
    ra, dec, radius_deg = _validate_sky(ra, dec, radius_deg)
    return f"""
SELECT TOP {int(limit)} source_id, ra, dec, pmra, pmdec,
       parallax, phot_g_mean_mag, ruwe, astrometric_excess_noise
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
  AND phot_g_mean_mag < {_num(g_max, "g_max")}
""".strip()


def adql_spherex_cutouts(ra: float, dec: float, size_deg: float,
                         limit: int = 1000) -> str:
    """Official cutout recipe: artifact JOIN plane, string-built URL."""
    ra, dec, size_deg = _validate_sky(ra, dec, size_deg)
    return f"""
SELECT TOP {int(limit)} 'https://irsa.ipac.caltech.edu/' || a.uri
       || '?center={ra},{dec}d&size={size_deg}' AS uri,
       p.time_bounds_lower, p.energy_bandpassname
FROM spherex.artifact a
JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
ORDER BY p.time_bounds_lower
""".strip()


def adql_new_planes(since_iso: str, band: str = "SPHEREx-D2", limit: int = 1000) -> str:
    """Find newly ingested planes after a timestamp. `since_iso` is not SQL."""
    if any(ch in since_iso for ch in "'\";"):
        raise ValueError("since_iso must be a plain timestamp, e.g. 2026-09-01T00:00:00Z")
    if not band.startswith("SPHEREx-D"):
        raise ValueError("band must look like SPHEREx-D2")
    return f"""
SELECT TOP {int(limit)} planeid, obsid, energy_bandpassname,
       time_bounds_lower, time_bounds_upper, poly
FROM spherex.plane
WHERE time_bounds_lower > '{since_iso}'
  AND energy_bandpassname = '{band}'
ORDER BY time_bounds_lower
""".strip()


# ---------------------------------------------------------------------------
# Coordinates, proper motion and matching
# ---------------------------------------------------------------------------

def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation, robust at small separations."""
    lon1, lat1 = math.radians(ra1), math.radians(dec1)
    lon2, lat2 = math.radians(ra2), math.radians(dec2)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def apply_proper_motion(ra: float, dec: float, pmra_mas_yr: float,
                        pmdec_mas_yr: float, mjd_obs: float,
                        epoch_ref: float = 2016.0) -> tuple[float, float]:
    """Approximate PM propagation; pmra is mu_alpha* (mas/yr).

    For high-PM nearby stars use astropy SkyCoord.apply_space_motion with
    distance/radial velocity instead.
    """
    dt_yr = ((float(mjd_obs) - 51544.5) / 365.25 + 2000.0) - float(epoch_ref)
    cos_dec = max(math.cos(math.radians(dec)), 1e-8)
    dra = pmra_mas_yr * dt_yr / 3.6e6 / cos_dec
    ddec = pmdec_mas_yr * dt_yr / 3.6e6
    return ra + dra, dec + ddec


def match_catalog(ra: float, dec: float, cat_ra: Sequence[float],
                  cat_dec: Sequence[float], radius_arcsec: float,
                  cat_pmra: Optional[Sequence[float]] = None,
                  cat_pmdec: Optional[Sequence[float]] = None,
                  mjd_obs: Optional[float] = None) -> tuple[Optional[int], float]:
    """Return (index, separation). If no match, index is None."""
    if len(cat_ra) != len(cat_dec):
        raise ValueError("catalog coordinate arrays differ in length")
    best_i: Optional[int] = None
    best_sep = float("inf")
    for i, (c_ra, c_dec) in enumerate(zip(cat_ra, cat_dec)):
        if cat_pmra is not None and cat_pmdec is not None and mjd_obs is not None:
            c_ra, c_dec = apply_proper_motion(c_ra, c_dec, cat_pmra[i], cat_pmdec[i], mjd_obs)
        sep = angular_separation_arcsec(ra, dec, c_ra, c_dec)
        if sep < best_sep:
            best_i, best_sep = i, sep
    if best_sep > radius_arcsec:
        return None, best_sep
    return best_i, best_sep


# ---------------------------------------------------------------------------
# Candidate classification
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    ra: float
    dec: float
    snr: float
    epochs: int
    catalog_match: Optional[str] = None       # "gaia" | "wise" | None
    bad_flags: tuple[str, ...] = ()
    psf_consistency: float = 1.0
    negative_ratio: float = 0.0
    edge_distance_psf: float = 10.0
    motion_arcsec: float = 0.0
    score: float = 0.0
    status: str = "transient"
    reasons: tuple[str, ...] = ()


def classify_candidate(c: Candidate) -> Candidate:
    reasons: list[str] = []
    if c.bad_flags:
        return Candidate(**{**c.__dict__, "status": "artifact",
                            "reasons": ("bad flag in aperture: " + ", ".join(c.bad_flags),)})
    if c.negative_ratio > 0.3:
        return Candidate(**{**c.__dict__, "status": "rejected",
                            "reasons": ("negative lobe too strong",)})
    if c.edge_distance_psf < 1.5:
        return Candidate(**{**c.__dict__, "status": "rejected",
                            "reasons": ("too close to the edge",)})
    if c.snr < 5:
        return Candidate(**{**c.__dict__, "status": "rejected",
                            "reasons": ("SNR below 5",)})
    if c.catalog_match == "gaia":
        return Candidate(**{**c.__dict__, "status": "known_star",
                            "reasons": ("Gaia counterpart after PM correction",)})
    if c.catalog_match == "wise":
        return Candidate(**{**c.__dict__, "status": "known_galaxy",
                            "reasons": ("WISE counterpart; no Gaia match",)})
    if c.epochs < 2:
        return Candidate(**{**c.__dict__, "status": "transient",
                            "reasons": ("single-epoch detection; needs confirmation",)})
    reasons.append("no catalog counterpart; detected in >=2 epochs")
    return Candidate(**{**c.__dict__, "status": "new_candidate", "reasons": tuple(reasons)})


def score_candidate(c: Candidate) -> float:
    catalog_penalty = 1.0 if not c.catalog_match else 0.0
    bad_fraction = 1.0 if not c.bad_flags else 0.0
    score = (0.35 * min(c.snr / 10.0, 1.0)
             + 0.20 * min(c.epochs / 3.0, 1.0)
             + 0.20 * catalog_penalty
             + 0.15 * max(0.0, min(c.psf_consistency, 1.0))
             + 0.10 * bad_fraction)
    return round(max(0.0, min(score, 1.0)), 4)


def to_manifest_candidate(c: Candidate, candidate_id: str) -> dict[str, Any]:
    c = classify_candidate(c)
    c.score = score_candidate(c)
    return {
        "id": candidate_id,
        "ra": round(c.ra, 8),
        "dec": round(c.dec, 8),
        "status": c.status,
        "snr": round(c.snr, 3),
        "epochs": int(c.epochs),
        "flags": list(c.bad_flags),
        "catalog_match": c.catalog_match,
        "motion_arcsec": round(c.motion_arcsec, 3),
        "score": c.score,
        "reasons": list(c.reasons),
    }


# ---------------------------------------------------------------------------
# Minimal manifest validation (no jsonschema dependency)
# ---------------------------------------------------------------------------

def validate_manifest(doc: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if doc.get("schema") != 1:
        errors.append("schema must be 1")
    if not doc.get("release"):
        errors.append("release is required")
    if not doc.get("generated_at"):
        errors.append("generated_at is required")
    for i, field in enumerate(doc.get("fields", [])):
        prefix = f"fields[{i}]"
        for key in ("id", "ra", "dec", "product", "qc", "catalog", "candidates"):
            if key not in field:
                errors.append(f"{prefix}.{key} is missing")
        product = field.get("product", {})
        detector = product.get("detector")
        if not isinstance(detector, int) or not 1 <= detector <= 6:
            errors.append(f"{prefix}.product.detector must be 1..6")
        if "FLAGS" not in product.get("hdus", []):
            errors.append(f"{prefix}.product.hdus must include FLAGS")
        for j, cand in enumerate(field.get("candidates", [])):
            if cand.get("status") not in {
                "new_candidate", "known_star", "known_galaxy",
                "artifact", "transient", "rejected",
            }:
                errors.append(f"{prefix}.candidates[{j}].status invalid")
    return errors


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    assert decode_flags(0) == []
    assert decode_flags(1) == ["TRANSIENT"]
    assert decode_flags(1 | (1 << 21)) == ["TRANSIENT", "SOURCE"]
    assert decode_flags((1 << 32) | (1 << 33)) == ["CONTAINS_BAD_PIXEL", "FIT_ERROR"]
    assert decode_flags(-1)[:3] == ["TRANSIENT", "OVERFLOW", "SUR_ERROR"]

    assert band_index(1, 1) == 1
    assert band_index(1, 17) == 17
    assert band_index(2, 1) == 18
    assert band_index(6, 17) == 102
    assert detector_channel(18) == (2, 1)
    assert detector_channel(102) == (6, 17)
    for idx in range(1, 103):
        assert band_index(*detector_channel(idx)) == idx
    for bad_ch in (0, 18):
        try:
            band_index(1, bad_ch)
        except ValueError:
            pass
        else:
            raise AssertionError("channel 0/18 must be rejected")

    assert abs(angular_separation_arcsec(10.0, 20.0, 10.0, 20.0)) < 1e-9
    sep = angular_separation_arcsec(10.0, 20.0, 10.0, 20.0 + 1 / 3600)
    assert abs(sep - 1.0) < 1e-6

    ra0, dec0 = 210.0, 54.0
    ra_obs, dec_obs = apply_proper_motion(ra0, dec0, 100.0, 50.0, 60000.0)
    assert 0.5 < angular_separation_arcsec(ra0, dec0, ra_obs, dec_obs) < 3.0

    idx, sep = match_catalog(10.0, 20.0, [10.0, 10.1], [20.0, 20.1], 2.0)
    assert idx == 0 and sep < 1e-6
    idx, _ = match_catalog(10.0, 20.0, [10.1], [20.1], 2.0)
    assert idx is None

    c = Candidate(ra=10.0, dec=20.0, snr=9.0, epochs=2)
    assert classify_candidate(c).status == "new_candidate"
    assert classify_candidate(Candidate(ra=10, dec=20, snr=9, epochs=2,
                                        catalog_match="gaia")).status == "known_star"
    assert classify_candidate(Candidate(ra=10, dec=20, snr=9, epochs=1)).status == "transient"
    assert classify_candidate(Candidate(ra=10, dec=20, snr=9, epochs=2,
                                        bad_flags=("TRANSIENT",))).status == "artifact"

    doc = {"schema": 1, "release": "DR1-2026.45", "generated_at": "2026-11-14T00:00:00Z",
           "fields": [{"id": "x", "ra": 10, "dec": 20,
                       "product": {"detector": 2, "hdus": ["IMAGE", "FLAGS"]},
                       "qc": {}, "catalog": {}, "candidates": []}]}
    assert validate_manifest(doc) == []
    assert validate_manifest({**doc, "release": ""})
    print("selftest: ok (flags, bands, PM, matching, classification, manifest)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SPHEREx QC primitives")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="run dependency-free checks")
    sub.add_parser("flags", help="print the flag table as JSON")
    args = parser.parse_args(argv)
    if args.cmd == "flags":
        print(json.dumps({"bits": FLAG_BITS, "groups": MASK_GROUPS,
                          "bad_for_detection": BAD_FOR_DETECTION}, indent=2))
        return 0
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
