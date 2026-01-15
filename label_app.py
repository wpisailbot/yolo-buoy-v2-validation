import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# IMPORTANT (Windows): torch may fail to load (WinError 1114 / c10.dll) if PyQt6 is imported first.
# Import ultralytics/torch BEFORE any PyQt6 imports to avoid a DLL initialization conflict.
from ultralytics import YOLO

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


CSV_NAME = "labels.csv"


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def image_number_from_path(p: Path) -> Optional[int]:
    m = re.match(r"^(\d+)\.[^.]+$", p.name)
    if not m:
        return None
    return int(m.group(1))


@dataclass(frozen=True)
class InferenceResult:
    image_path: Path
    overlay_bgr: np.ndarray  # BGR image with boxes drawn
    det_count: int
    conf_max: float
    conf_avg: float


class WorkerSignals(QObject):
    finished = pyqtSignal(object)  # InferenceResult
    error = pyqtSignal(str)


class InferenceWorker(QObject):
    """
    Simple QObject worker executed by QThreadPool via QRunnable-like pattern.
    We keep it minimal and CPU-friendly; YOLO inference itself releases GIL in many ops.
    """

    def __init__(self, model: YOLO, image_path: Path, conf_thres: float, imgsz: int):
        super().__init__()
        self.signals = WorkerSignals()
        self.model = model
        self.image_path = image_path
        self.conf_thres = conf_thres
        self.imgsz = imgsz

    @pyqtSlot()
    def run(self):
        try:
            img_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read image: {self.image_path}")

            # Ultralytics expects RGB by default; passing file path is simplest and consistent.
            results = self.model.predict(
                source=str(self.image_path),
                conf=self.conf_thres,
                imgsz=self.imgsz,
                verbose=False,
                device="cpu",
            )
            r0 = results[0]

            boxes = getattr(r0, "boxes", None)
            det_count = 0
            confs: List[float] = []

            overlay = img_bgr.copy()
            if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None

                det_count = xyxy.shape[0]
                for i in range(det_count):
                    x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
                    c = float(conf[i]) if conf is not None else 0.0
                    confs.append(c)
                    label = f"{cls[i]}" if cls is not None else "buoy"
                    text = f"{label} {c:.2f}"

                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 220, 0), -1)
                    cv2.putText(
                        overlay,
                        text,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            conf_max = float(max(confs)) if confs else 0.0
            conf_avg = float(sum(confs) / len(confs)) if confs else 0.0

            out = InferenceResult(
                image_path=self.image_path,
                overlay_bgr=overlay,
                det_count=det_count,
                conf_max=conf_max,
                conf_avg=conf_avg,
            )
            self.signals.finished.emit(out)
        except Exception as e:
            self.signals.error.emit(str(e))


class RunnableAdapter(QObject):
    """
    QThreadPool expects QRunnable; easiest is to wrap a callable in QRunnable,
    but PyQt6 QRunnable doesn't support signals nicely without subclassing.
    We use QtConcurrent-like pattern: post a slot via QObject and use a QRunnable shim.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __call__(self):
        self.fn()


try:
    from PyQt6.QtCore import QRunnable
except Exception:  # pragma: no cover
    QRunnable = object  # type: ignore


class QRunnableShim(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class LabelStore:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.rows: Dict[int, Dict[str, float]] = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            return
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            n = int(row["image_number"])
            self.rows[n] = {
                "buoy_actual": float(row.get("buoy_actual", np.nan)),
                "buoy_detected": float(row.get("buoy_detected", np.nan)),
                "certainty_avg": float(row.get("certainty_avg", np.nan)),
            }

    def upsert(self, image_number: int, buoy_actual: float, buoy_detected: float, certainty_avg: float):
        self.rows[image_number] = {
            "buoy_actual": float(buoy_actual),
            "buoy_detected": float(buoy_detected),
            "certainty_avg": float(certainty_avg),
        }
        self._flush()

    def get(self, image_number: int) -> Optional[Dict[str, float]]:
        return self.rows.get(image_number)

    def _flush(self):
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["image_number", "buoy_actual", "buoy_detected", "certainty_avg"],
            )
            w.writeheader()
            for image_number in sorted(self.rows.keys()):
                row = self.rows[image_number]
                w.writerow(
                    {
                        "image_number": image_number,
                        "buoy_actual": row["buoy_actual"],
                        "buoy_detected": row["buoy_detected"],
                        "certainty_avg": row["certainty_avg"],
                    }
                )
        tmp.replace(self.csv_path)


class MainWindow(QMainWindow):
    def __init__(self, repo_root: Path):
        super().__init__()
        self.repo_root = repo_root
        self.images_dir = repo_root / "images"
        self.model_path = repo_root / "my_model.pt"
        self.csv_path = repo_root / CSV_NAME

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")

        self.setWindowTitle("YOLO Buoy Validation")
        self.resize(1300, 850)

        self.store = LabelStore(self.csv_path)
        self.model = YOLO(str(self.model_path))

        self.images: List[Path] = list_images(self.images_dir)
        if not self.images:
            raise RuntimeError("No images found in images/.")

        self.idx = 0
        self.last_infer: Optional[InferenceResult] = None
        self._current_pixmap_original: Optional[QPixmap] = None

        # Fixed model inference params (kept out of the UI to avoid confusing them with "certainty").
        self.model_conf_thres = 0.25
        self.model_imgsz = 640

        self.thread_pool = QThreadPool.globalInstance()

        self._build_ui()
        self._install_shortcuts()

        self.load_image(self.idx)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)

        # Left: image preview
        self.image_label = QLabel("Loading…")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #333;")
        # Prevent "infinite growth" feedback loop where pixmap sizeHint expands the layout, which then
        # causes us to scale a larger pixmap, which expands sizeHint again.
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        outer.addWidget(self.image_label, stretch=3)

        # Right: controls
        right = QVBoxLayout()
        outer.addLayout(right, stretch=2)

        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        right.addWidget(self.meta_label)

        label_group = QGroupBox("Label (Ground Truth)")
        label_layout = QVBoxLayout(label_group)

        btn_row = QHBoxLayout()
        self.btn_yes = QPushButton("Buoy present (1)  [A]")
        self.btn_no = QPushButton("No buoy (0)  [S]")
        self.btn_yes.clicked.connect(lambda: self.set_actual_and_save(1.0))
        self.btn_no.clicked.connect(lambda: self.set_actual_and_save(0.0))
        btn_row.addWidget(self.btn_yes)
        btn_row.addWidget(self.btn_no)
        label_layout.addLayout(btn_row)

        edge_row = QHBoxLayout()
        self.edge_value = QDoubleSpinBox()
        self.edge_value.setRange(0.0, 1.0)
        self.edge_value.setSingleStep(0.05)
        self.edge_value.setValue(0.5)
        self.btn_edge = QPushButton("Use edge-case decimal  [D]")
        self.btn_edge.clicked.connect(lambda: self.set_actual_and_save(float(self.edge_value.value())))
        edge_row.addWidget(QLabel("Ground truth (edge-case decimal):"))
        edge_row.addWidget(self.edge_value)
        edge_row.addWidget(self.btn_edge)
        label_layout.addLayout(edge_row)

        self.current_label = QLabel("Saved label: (none)")
        self.current_label.setWordWrap(True)
        label_layout.addWidget(self.current_label)

        right.addWidget(label_group)

        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        self.btn_prev = QPushButton("← Prev")
        self.btn_next = QPushButton("Next →")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        right.addWidget(nav_group)

        misc_group = QGroupBox("Misc")
        misc_layout = QHBoxLayout(misc_group)
        self.btn_open_images = QPushButton("Choose images folder…")
        self.btn_open_images.clicked.connect(self.choose_images_folder)
        misc_layout.addWidget(self.btn_open_images)
        right.addWidget(misc_group)

        right.addStretch(1)

    def _install_shortcuts(self):
        # Menu actions for shortcuts
        act_prev = QAction(self)
        act_prev.setShortcut(QKeySequence(Qt.Key.Key_Left))
        act_prev.triggered.connect(self.prev_image)
        self.addAction(act_prev)

        act_next = QAction(self)
        act_next.setShortcut(QKeySequence(Qt.Key.Key_Right))
        act_next.triggered.connect(self.next_image)
        self.addAction(act_next)

        act_yes = QAction(self)
        act_yes.setShortcut(QKeySequence("A"))
        act_yes.triggered.connect(lambda: self.set_actual_and_save(1.0))
        self.addAction(act_yes)

        act_no = QAction(self)
        act_no.setShortcut(QKeySequence("S"))
        act_no.triggered.connect(lambda: self.set_actual_and_save(0.0))
        self.addAction(act_no)

        act_edge = QAction(self)
        act_edge.setShortcut(QKeySequence("D"))
        act_edge.triggered.connect(lambda: self.set_actual_and_save(float(self.edge_value.value())))
        self.addAction(act_edge)

        act_save = QAction(self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_current_again)
        self.addAction(act_save)

    def choose_images_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select images folder", str(self.images_dir))
        if not d:
            return
        new_dir = Path(d)
        imgs = list_images(new_dir)
        if not imgs:
            QMessageBox.warning(self, "No images", "That folder contains no supported image files.")
            return
        self.images_dir = new_dir
        self.images = imgs
        self.idx = 0
        self.load_image(self.idx)

    def _status_text(self) -> str:
        p = self.images[self.idx]
        n = image_number_from_path(p)
        n_txt = str(n) if n is not None else "(non-numeric filename)"
        saved = self.store.get(n) if n is not None else None

        parts = [
            f"Image: {p.name}  ({self.idx + 1}/{len(self.images)})",
            f"Image number: {n_txt}",
        ]
        if self.last_infer is not None and self.last_infer.image_path == p:
            parts.append(f"Detections: {self.last_infer.det_count}")
            parts.append(f"Detected: {1 if self.last_infer.det_count > 0 else 0}")
            parts.append(f"Certainty avg: {self.last_infer.conf_avg:.3f}")
        else:
            parts.append("Detections: (running…)")

        if saved is not None:
            parts.append(
                f"Saved label — buoy_actual={saved['buoy_actual']}, "
                f"buoy_detected={saved['buoy_detected']}, "
                f"certainty_avg={saved['certainty_avg']}"
            )
        return "\n".join(parts)

    def load_image(self, idx: int):
        self.idx = max(0, min(idx, len(self.images) - 1))
        self.last_infer = None
        self.meta_label.setText(self._status_text())
        self.current_label.setText("Saved label: (none)" if self._saved_for_current() is None else "Saved label: present")

        # Show raw image quickly while inference runs
        p = self.images[self.idx]
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            self.image_label.setText(f"Failed to load {p.name}")
            return
        self._set_image(img_bgr)
        self._run_inference(p)

    def _saved_for_current(self) -> Optional[Dict[str, float]]:
        n = image_number_from_path(self.images[self.idx])
        if n is None:
            return None
        return self.store.get(n)

    def _set_image(self, bgr: np.ndarray):
        # Convert BGR to RGB QImage
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_pixmap_original = QPixmap.fromImage(qimg)
        self._refresh_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale current pixmap on window resize without changing the pixmap "source".
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self):
        if self._current_pixmap_original is None:
            return
        target = self.image_label.size()
        if target.width() <= 1 or target.height() <= 1:
            return
        scaled = self._current_pixmap_original.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _rerun_inference(self):
        self._run_inference(self.images[self.idx])

    def _run_inference(self, image_path: Path):
        worker = InferenceWorker(
            model=self.model,
            image_path=image_path,
            conf_thres=float(self.model_conf_thres),
            imgsz=int(self.model_imgsz),
        )
        worker.signals.finished.connect(self._on_infer_done)
        worker.signals.error.connect(self._on_infer_error)

        # schedule in thread pool
        runnable = QRunnableShim(worker.run)
        self.thread_pool.start(runnable)

    @pyqtSlot(object)
    def _on_infer_done(self, res_obj):
        res: InferenceResult = res_obj
        # Ignore stale results
        if res.image_path != self.images[self.idx]:
            return
        self.last_infer = res
        self._set_image(res.overlay_bgr)
        self.meta_label.setText(self._status_text())

    @pyqtSlot(str)
    def _on_infer_error(self, msg: str):
        self.meta_label.setText(self._status_text() + f"\n\nInference error: {msg}")

    def prev_image(self):
        if self.idx > 0:
            self.load_image(self.idx - 1)

    def next_image(self):
        if self.idx < len(self.images) - 1:
            self.load_image(self.idx + 1)

    def set_actual_and_save(self, buoy_actual: float):
        p = self.images[self.idx]
        n = image_number_from_path(p)
        if n is None:
            QMessageBox.warning(
                self,
                "Non-numeric filename",
                f"File {p.name} isn't named like '12.jpg'. Run prepare_images.py first.",
            )
            return

        buoy_detected = 1.0 if (self.last_infer is not None and self.last_infer.det_count > 0) else 0.0
        certainty_avg = float(self.last_infer.conf_avg) if self.last_infer is not None else 0.0

        self.store.upsert(
            image_number=n,
            buoy_actual=float(buoy_actual),
            buoy_detected=buoy_detected,
            certainty_avg=certainty_avg,
        )

        self.current_label.setText(
            f"Saved label: image_number={n}, buoy_actual={buoy_actual}, buoy_detected={int(buoy_detected)}, certainty_avg={certainty_avg:.3f}"
        )
        self.meta_label.setText(self._status_text())

        # Common workflow: auto-advance after 0/1, but NOT for edge-case decimal.
        if buoy_actual in (0.0, 1.0):
            self.next_image()

    def save_current_again(self):
        saved = self._saved_for_current()
        if saved is None:
            QMessageBox.information(self, "Nothing to save", "No saved label for this image yet. Use A/S/D first.")
            return
        p = self.images[self.idx]
        n = image_number_from_path(p)
        if n is None:
            return
        buoy_actual = float(saved["buoy_actual"])
        self.set_actual_and_save(buoy_actual)


def main():
    repo_root = Path(__file__).resolve().parent
    app = QApplication(sys.argv)
    win = MainWindow(repo_root)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


