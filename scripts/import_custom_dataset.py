#!/usr/bin/env python3
"""Import selected frames from a hand-collected dataset into VidBot dataset folders.

This script reads:
  - dataset.xlsx      : clip names, selected frame indices, instructions, calibration IDs
  - calibration.xlsx  : mapping of calibration IDs → source clip folders containing
                        calibration.json files produced by estimate_instrinsics.py

For each clip it creates:
    datasets/<clip>/
      color/000000.png        — selected frame (copied or symlinked)
      camera_intrinsic.json   — intrinsics resolved from the clip's calibration ID,
                                scaled to the actual frame resolution if needed
      selection.json          — full provenance metadata
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET


# ── XLSX parsing ───────────────────────────────────────────────────────────────

XML_NS = {
    "a":  "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r":  "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _col_to_num(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch.upper()) - 64)
    return value


def _split_cell_ref(cell_ref: str) -> tuple[str, int | None]:
    col, row = "", ""
    for ch in cell_ref:
        (col if ch.isalpha() else row).__iadd__ if False else None
        if ch.isalpha():
            col += ch
        elif ch.isdigit():
            row += ch
    return col, int(row) if row else None


def read_xlsx_rows(path: Path) -> list[list[str]]:
    """Return all non-empty rows of the first sheet as lists of strings."""
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", XML_NS):
                texts = [t.text or "" for t in
                         si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                shared_strings.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels     = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map  = {rel.attrib["Id"]: rel.attrib["Target"]
                    for rel in rels.findall("pr:Relationship", XML_NS)}
        first_sheet = workbook.find("a:sheets", XML_NS)[0]
        rel_id      = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        sheet_path  = Path("xl") / rel_map[rel_id]
        sheet       = ET.fromstring(zf.read(str(sheet_path)))

        rows: list[list[str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", XML_NS):
            values_by_col: dict[int, str] = {}
            for cell in row.findall("a:c", XML_NS):
                ref        = cell.attrib.get("r", "")
                col_letters, _ = _split_cell_ref(ref)
                col_idx    = _col_to_num(col_letters)
                cell_type  = cell.attrib.get("t")
                value_node = cell.find("a:v", XML_NS)
                inline_node = cell.find("a:is", XML_NS)
                value = ""
                if cell_type == "s" and value_node is not None and value_node.text is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(
                        t.text or "" for t in
                        inline_node.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    )
                elif value_node is not None and value_node.text is not None:
                    value = value_node.text
                values_by_col[col_idx] = value

            if values_by_col:
                max_col = max(values_by_col)
                rows.append([values_by_col.get(i, "") for i in range(1, max_col + 1)])

    return rows


# ── data models ────────────────────────────────────────────────────────────────

@dataclass
class DatasetRow:
    video:        str
    frame_idx:    int
    twist:        int
    force_level:  int
    instruction:  str
    calibration:  str   # calibration ID, resolved via calibration.xlsx
    author:       str


@dataclass
class CalibrationEntry:
    calib_id:      str
    source_folder: str          # folder inside source_root containing calibration.json
    square_size_mm: float = 0.0


# ── parsers ────────────────────────────────────────────────────────────────────

def parse_dataset_rows(path: Path) -> list[DatasetRow]:
    rows = read_xlsx_rows(path)
    out: list[DatasetRow] = []
    for raw in rows[1:]:           # skip header
        if not raw or not raw[0]:
            continue
        out.append(DatasetRow(
            video        = raw[0],
            frame_idx    = int(float(raw[1])),
            twist        = int(float(raw[2])),
            force_level  = int(float(raw[3])),
            instruction  = raw[4],
            calibration  = raw[5],
            author       = raw[6] if len(raw) > 6 else "",
        ))
    return out


def parse_calibration_table(path: Path) -> dict[str, CalibrationEntry]:
    """Parse calibration.xlsx into {calib_id: CalibrationEntry}."""
    rows = read_xlsx_rows(path)
    out: dict[str, CalibrationEntry] = {}
    for raw in rows[1:]:           # skip header
        if not raw or not raw[0]:
            continue
        calib_id      = raw[0].strip()
        source_folder = raw[1].strip() if len(raw) > 1 else ""
        square_size   = float(raw[2]) if len(raw) > 2 and raw[2] else 0.0
        if calib_id and source_folder:
            out[calib_id] = CalibrationEntry(
                calib_id       = calib_id,
                source_folder  = source_folder,
                square_size_mm = square_size,
            )
    return out


# ── intrinsics ────────────────────────────────────────────────────────────────

def load_calibration_json(calib_json_path: Path) -> dict:
    with calib_json_path.open("r") as f:
        return json.load(f)


def build_camera_intrinsic(
    calib: dict,
    actual_width:  int,
    actual_height: int,
) -> dict:
    """
    Build the VidBot camera_intrinsic.json dict from a calibration.json,
    scaling fx/fy/cx/cy if the actual frame resolution differs from the
    resolution the calibration was computed at.
    """
    K           = calib["camera_matrix"]
    fx_cal      = float(K[0][0])
    fy_cal      = float(K[1][1])
    cx_cal      = float(K[0][2])
    cy_cal      = float(K[1][2])
    cal_w, cal_h = calib["image_size_px"]   # resolution used during calibration

    # Scale intrinsics if frame resolution differs from calibration resolution
    sx = actual_width  / cal_w
    sy = actual_height / cal_h
    if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
        print(f"    ⚠ frame {actual_width}×{actual_height} vs calib {cal_w}×{cal_h} "
              f"— scaling intrinsics by ({sx:.4f}, {sy:.4f})")
    fx = fx_cal * sx
    fy = fy_cal * sy
    cx = cx_cal * sx
    cy = cy_cal * sy

    return {
        "width":  actual_width,
        "height": actual_height,
        # VidBot convention: flattened, transposed on load (reshape(3,3).T)
        "intrinsic_matrix": [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0],
    }


# ── frame selection ───────────────────────────────────────────────────────────

def choose_source_frame(video_dir: Path, frame_idx: int) -> tuple[Path, int, int]:
    frames = sorted(p for p in video_dir.glob("*.png") if p.is_file())
    if not frames:
        raise FileNotFoundError(f"No PNG frames in {video_dir}")
    resolved = frame_idx if frame_idx >= 0 else len(frames) + frame_idx
    resolved = max(0, min(resolved, len(frames) - 1))
    return frames[resolved], resolved, len(frames)


def frame_resolution(frame_path: Path) -> tuple[int, int]:
    """Return (width, height) without importing heavy deps — use struct parsing."""
    # PNG: width/height at bytes 16-23 in the IHDR chunk
    with frame_path.open("rb") as f:
        f.seek(16)
        import struct
        w, h = struct.unpack(">II", f.read(8))
    return int(w), int(h)


# ── helpers ───────────────────────────────────────────────────────────────────

def instruction_to_object(instruction: str) -> str:
    instr = instruction.strip().lower()
    if instr.startswith("open the "):
        return instr[len("open the "):]
    if instr.startswith("press the "):
        return instr[len("press the "):]
    if instr.startswith("pull the ") and instr.endswith(" closer"):
        return instr[len("pull the "):-len(" closer")]
    if instr.startswith("turn the ") and " to " in instr:
        return instr[len("turn the "):instr.index(" to ")]
    if instr.startswith("turn on the "):
        return instr[len("turn on the "):]
    return instr


def prepare_target_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists (use --overwrite): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="/Users/davidkorcak/Desktop/dataset",
        help="Desktop dataset root (contains dataset.xlsx, calibration.xlsx, clip folders).",
    )
    parser.add_argument(
        "--vidbot-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="VidBot repository root (datasets/ will be written here).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy frames instead of symlinking.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset folders.",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    vidbot_root = Path(args.vidbot_root).expanduser().resolve()
    target_root = vidbot_root / "datasets"

    dataset_xlsx     = source_root / "dataset.xlsx"
    calibration_xlsx = source_root / "calibration.xlsx"

    if not dataset_xlsx.exists():
        raise FileNotFoundError(f"Missing dataset sheet: {dataset_xlsx}")
    if not calibration_xlsx.exists():
        raise FileNotFoundError(f"Missing calibration sheet: {calibration_xlsx}")

    # Build calibration lookup: id → CalibrationEntry
    calib_table = parse_calibration_table(calibration_xlsx)
    print(f"Loaded {len(calib_table)} calibration(s): {list(calib_table.keys())}")

    rows    = parse_dataset_rows(dataset_xlsx)
    summary: list[dict] = []
    errors:  list[str]  = []

    for row in rows:
        print(f"\n{row.video}  (frame={row.frame_idx}, calib={row.calibration})")

        # ── Resolve calibration ──────────────────────────────────────────────
        if row.calibration not in calib_table:
            msg = (f"  ✗ Unknown calibration ID '{row.calibration}' "
                   f"(available: {list(calib_table.keys())})")
            print(msg); errors.append(msg); continue

        entry       = calib_table[row.calibration]
        calib_path  = source_root / entry.source_folder / "calibration.json"
        if not calib_path.exists():
            msg = f"  ✗ calibration.json not found: {calib_path}"
            print(msg); errors.append(msg); continue

        calib_data = load_calibration_json(calib_path)
        print(f"  calibration: {calib_path.relative_to(source_root)}"
              f"  rms={calib_data.get('rms_error', '?'):.4f}")

        # ── Source frame ─────────────────────────────────────────────────────
        video_dir = source_root / row.video
        if not video_dir.exists():
            msg = f"  ✗ Video folder not found: {video_dir}"
            print(msg); errors.append(msg); continue

        try:
            src_frame, resolved_idx, total = choose_source_frame(video_dir, row.frame_idx)
        except FileNotFoundError as e:
            print(f"  ✗ {e}"); errors.append(str(e)); continue

        # ── Actual frame resolution → scale intrinsics if needed ─────────────
        actual_w, actual_h = frame_resolution(src_frame)
        camera_intrinsic   = build_camera_intrinsic(calib_data, actual_w, actual_h)

        # ── Write dataset folder ──────────────────────────────────────────────
        dataset_dir = target_root / row.video
        color_dir   = dataset_dir / "color"
        try:
            prepare_target_dir(dataset_dir, overwrite=args.overwrite)
        except FileExistsError as e:
            print(f"  ✗ {e}"); errors.append(str(e)); continue
        color_dir.mkdir(parents=True, exist_ok=True)

        target_frame = color_dir / "000000.png"
        link_or_copy(src_frame, target_frame, copy_files=args.copy)

        with (dataset_dir / "camera_intrinsic.json").open("w") as f:
            json.dump(camera_intrinsic, f, indent=2)

        selection = {
            "video":               row.video,
            "instruction":         row.instruction,
            "object":              instruction_to_object(row.instruction),
            "author":              row.author,
            "calibration":         row.calibration,
            "calibration_source":  str(calib_path),
            "calibration_rms":     calib_data.get("rms_error"),
            "twist":               row.twist,
            "force_level":         row.force_level,
            "source_video_dir":    str(video_dir),
            "source_frame_path":   str(src_frame),
            "source_frame_idx":    resolved_idx,
            "frame_idx_from_sheet": row.frame_idx,
            "num_source_frames":   total,
            "frame_width":         actual_w,
            "frame_height":        actual_h,
            "prepared_color_frame": str(target_frame),
        }
        with (dataset_dir / "selection.json").open("w") as f:
            json.dump(selection, f, indent=2)

        summary.append(selection)
        print(f"  ✓ frame {resolved_idx}/{total - 1}  "
              f"{actual_w}×{actual_h}  → {target_frame}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = target_root / "desktop_dataset_import_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Imported {len(summary)}/{len(rows)} clips → {target_root}")
    if errors:
        print(f"{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
