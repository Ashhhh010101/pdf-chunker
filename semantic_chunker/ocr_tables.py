"""Recover ruled scan tables using raster grid geometry and OCR line coordinates.

Merged cells are assigned to their upper/left logical grid cell. The original
page/row boxes and empty cells remain available for auditing.
"""
from __future__ import annotations

import fitz
import numpy as np

from .models import Element, stable_id


def _centers(values, tolerance=4):
    groups = []
    for value in values:
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(g) / len(g) for g in groups]


def scan_tables(page, blocks, doc_id, method):
    import cv2
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Deskew scan grids before morphology; even a one-degree tilt breaks long kernels.
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=80,
                           minLineLength=pix.width // 4, maxLineGap=20)
    angles = []
    for line in [] if lines is None else lines[:, 0]:
        x1, y1, x2, y2 = line
        if abs(x2-x1) > pix.width / 4:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) < 5:
                angles.append(angle)
    angle = float(np.median(angles)) if angles else 0.0
    transform = cv2.getRotationMatrix2D((pix.width/2, pix.height/2), angle, 1)
    inverse = cv2.invertAffineTransform(transform)
    gray = cv2.warpAffine(gray, transform, (pix.width, pix.height), borderValue=255)

    def source_box(box):
        x0, y0, x1, y1 = box
        corners = np.array([[x0*2, y0*2, 1], [x1*2, y0*2, 1],
                            [x0*2, y1*2, 1], [x1*2, y1*2, 1]]) @ inverse.T / 2
        return [float(corners[:, 0].min()), float(corners[:, 1].min()),
                float(corners[:, 0].max()), float(corners[:, 1].max())]
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, pix.width // 30), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, pix.height // 40))))
    grid = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < pix.width * .3 or h < 80:
            continue
        xs = _centers(np.flatnonzero((vertical[y:y+h, x:x+w] > 0).sum(axis=0) > h * .2))
        ys = _centers(np.flatnonzero((horizontal[y:y+h, x:x+w] > 0).sum(axis=1) > w * .35))
        if len(xs) < 3 or len(ys) < 3:
            continue
        xs, ys = [(x+a)/2 for a in xs], [(y+a)/2 for a in ys]
        rows = [[[] for _ in range(len(xs)-1)] for _ in range(len(ys)-1)]
        confidences = []
        for block in blocks:
            for line in block.get("lines", []):
                box = line["bbox"]
                cx, cy = (transform @ np.array([box[0]+box[2], box[1]+box[3], 1])) / 2
                if not xs[0] <= cx <= xs[-1] or not ys[0] <= cy <= ys[-1]:
                    continue
                r = min(len(rows)-1, int(np.searchsorted(ys, cy, side="right"))-1)
                c = min(len(xs)-2, int(np.searchsorted(xs, cx, side="right"))-1)
                rows[r][c].append((box[1], box[0], "".join(s["text"] for s in line["spans"])))
                confidences.append(float(block.get("confidence", 1.0)))
        cells = [["\n".join(t[2] for t in sorted(cell)) for cell in row] for row in rows]
        if sum(sum(bool(c) for c in row) >= 2 for row in cells) < 2:
            continue
        bbox = source_box([xs[0], ys[0], xs[-1], ys[-1]])
        tid = stable_id(doc_id, page.number+1, "ocr_table", bbox)
        tables.append(Element(tid, page.number+1, bbox, "table", "\n".join(" | ".join(r) for r in cells),
                              rows=cells, row_boxes=[source_box([xs[0], ys[i], xs[-1], ys[i+1]]) for i in range(len(cells))],
                              header=[f"Column {i+1}" for i in range(len(xs)-1)], table_id=tid, extraction=method,
                              confidence=min(confidences) if confidences else 0))
    return tables
