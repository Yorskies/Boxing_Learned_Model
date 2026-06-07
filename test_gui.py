"""
Boxing Punch Classifier – Testing GUI
======================================
Comprehensive Tkinter GUI for testing trained AttentionBiLSTM models.
Supports single file inference, batch folder testing, ground-truth
comparison, confusion matrix generation, and CSV export.
"""

import os
import sys
import csv
import json
import time
import threading
import datetime
from collections import Counter
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src', 'preprocessing'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src', 'training'))

from model import AttentionBiLSTM
from video_processor import get_video_frames
from mediapipe_extractor import extract_keypoints, compute_engineered_features
from sequence_padder import pad_or_truncate_sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES = ['Hook', 'Jab', 'Straight', 'Uppercut']
CLASS_COLORS = {
    'Hook':      '#e74c3c',
    'Jab':       '#3498db',
    'Straight':  '#2ecc71',
    'Uppercut':  '#f39c12',
}
# BGR colours for OpenCV rendering
CLASS_COLORS_BGR = {
    'Hook':      (60, 76, 231),
    'Jab':       (219, 152, 52),
    'Straight':  (113, 204, 46),
    'Uppercut':  (18, 156, 243),
}

SCENARIOS = {
    1: {"name": "Skenario 1: Baseline (hidden=64, layers=1)",
        "hidden_size": 64, "num_layers": 1, "dropout": 0.3},
    2: {"name": "Skenario 2: Deep & Regularized (hidden=128, layers=2)",
        "hidden_size": 128, "num_layers": 2, "dropout": 0.5},
    3: {"name": "Skenario 3: Lightweight (hidden=32, layers=1)",
        "hidden_size": 32, "num_layers": 1, "dropout": 0.2},
    4: {"name": "Skenario 4: Ultimate (BS=16)",
        "hidden_size": 64, "num_layers": 1, "dropout": 0.3},
    5: {"name": "Skenario 5: Ultimate (BS=4) [FINAL]",
        "hidden_size": 64, "num_layers": 1, "dropout": 0.3},
}

SUPPORTED_VIDEO = ('.mp4', '.avi', '.mov', '.mkv')
SUPPORTED_NUMPY = ('.npy',)
SUPPORTED_ALL   = SUPPORTED_VIDEO + SUPPORTED_NUMPY

# MediaPipe Pose skeleton connections (same as render_keypoints.py)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
]

# Results source dropdown
RESULT_DIRS = []
for name in sorted(os.listdir(ROOT_DIR)):
    full = os.path.join(ROOT_DIR, name)
    if os.path.isdir(full) and name.startswith('results'):
        RESULT_DIRS.append(name)
if not RESULT_DIRS:
    RESULT_DIRS = ['results']


# ═══════════════════════════════════════════════════════════════════════════
class PunchClassifierGUI:
    """Main application class."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🥊 Boxing Punch Classifier – Test Suite")
        self.root.geometry("880x920")
        self.root.minsize(780, 800)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.current_scenario = None
        self.current_result_dir = RESULT_DIRS[-1]   # default latest

        # Batch results storage  [{file, true_label, pred_label, confidence, probs}, ...]
        self.batch_results: list[dict] = []
        self._is_running = False

        self._apply_theme()
        self._build_ui()
        self._load_initial_model()

    # ------------------------------------------------------------------
    # Theme / styles
    # ------------------------------------------------------------------
    def _apply_theme(self):
        s = ttk.Style()
        s.theme_use('clam')

        BG = "#f0f2f5"
        self.BG = BG
        self.root.configure(bg=BG)

        s.configure(".",           font=("Segoe UI", 10), background=BG)
        s.configure("TFrame",     background=BG)
        s.configure("TLabel",     background=BG, font=("Segoe UI", 10))
        s.configure("TLabelframe",       background=BG)
        s.configure("TLabelframe.Label", background=BG, font=("Segoe UI", 10, "bold"))
        s.configure("Header.TLabel",     font=("Segoe UI", 15, "bold"), foreground="#1a1a2e")
        s.configure("Sub.TLabel",        font=("Segoe UI", 9), foreground="#555")
        s.configure("Result.TLabel",     font=("Segoe UI", 14, "bold"), foreground="#16213e")
        s.configure("TButton",          padding=5, font=("Segoe UI", 9, "bold"))

        # Coloured progress bar styles (one per class)
        for cls, color in CLASS_COLORS.items():
            style_name = f"{cls}.Horizontal.TProgressbar"
            s.configure(style_name, troughcolor="#dfe6e9", background=color)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Scrollable canvas wrapper
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, bg=self.BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.main_frame = ttk.Frame(canvas, padding=20)
        canvas_window = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
        self.main_frame.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_configure)

        # Enable mouse-wheel scrolling
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        mf = self.main_frame

        # ── Title ──
        ttk.Label(mf, text="🥊 Boxing Punch Classifier – Test Suite",
                  style="Header.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(mf, text="Uji model klasifikasi pukulan tinju pada file atau folder secara menyeluruh.",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 12))

        # ── Section 1: Model Configuration ──
        sec1 = ttk.LabelFrame(mf, text="1 · Konfigurasi Model", padding=12)
        sec1.pack(fill=tk.X, pady=(0, 8))

        row0 = ttk.Frame(sec1); row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="Sumber Hasil (Results Folder):").pack(side=tk.LEFT)
        self.result_dir_var = tk.StringVar(value=self.current_result_dir)
        cb_rd = ttk.Combobox(row0, textvariable=self.result_dir_var,
                             values=RESULT_DIRS, width=18, state="readonly")
        cb_rd.pack(side=tk.LEFT, padx=8)
        cb_rd.bind("<<ComboboxSelected>>", self._on_result_dir_change)

        row1 = ttk.Frame(sec1); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Skenario Model:").pack(side=tk.LEFT)
        self.scenario_var = tk.StringVar(value="5")
        cb_sc = ttk.Combobox(row1, textvariable=self.scenario_var,
                             values=["1", "2", "3", "4", "5"], width=4, state="readonly")
        cb_sc.pack(side=tk.LEFT, padx=8)
        cb_sc.bind("<<ComboboxSelected>>", self._on_scenario_change)

        self.scenario_desc_lbl = ttk.Label(row1, text=SCENARIOS[1]["name"],
                                           font=("Segoe UI", 9, "italic"), foreground="#555")
        self.scenario_desc_lbl.pack(side=tk.LEFT, padx=8)

        self.device_lbl = ttk.Label(sec1, text=f"⚡ Device: {self.device.type.upper()}",
                                    font=("Segoe UI", 9, "bold"), foreground="#27ae60")
        self.device_lbl.pack(anchor="e")

        # ── Section 2: Input ──
        sec2 = ttk.LabelFrame(mf, text="2 · Input", padding=12)
        sec2.pack(fill=tk.X, pady=(0, 8))

        # 2a – single file
        ttk.Label(sec2, text="File Tunggal (Video / .npy):").pack(anchor="w")
        row_file = ttk.Frame(sec2); row_file.pack(fill=tk.X, pady=2)
        self.file_path_var = tk.StringVar()
        ttk.Entry(row_file, textvariable=self.file_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_browse_file = ttk.Button(row_file, text="Browse File…", command=self._browse_file)
        self.btn_browse_file.pack(side=tk.LEFT)

        # 2b – folder (batch)
        ttk.Label(sec2, text="Folder Batch (Semua file di dalam folder):").pack(anchor="w", pady=(8, 0))
        row_folder = ttk.Frame(sec2); row_folder.pack(fill=tk.X, pady=2)
        self.folder_path_var = tk.StringVar()
        ttk.Entry(row_folder, textvariable=self.folder_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_browse_folder = ttk.Button(row_folder, text="Browse Folder…", command=self._browse_folder)
        self.btn_browse_folder.pack(side=tk.LEFT)

        # 2c – ground-truth label (optional)
        row_gt = ttk.Frame(sec2); row_gt.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row_gt, text="Ground Truth (Opsional):").pack(side=tk.LEFT)
        self.gt_var = tk.StringVar(value="Auto-detect")
        gt_cb = ttk.Combobox(row_gt, textvariable=self.gt_var,
                             values=["Auto-detect"] + CLASSES, width=14, state="readonly")
        gt_cb.pack(side=tk.LEFT, padx=8)
        ttk.Label(row_gt, text="(Auto = deteksi dari nama folder induk file)",
                  style="Sub.TLabel").pack(side=tk.LEFT)

        # ── Section 3: Actions ──
        sec3 = ttk.Frame(mf); sec3.pack(fill=tk.X, pady=6)

        self.btn_run_single = ttk.Button(sec3, text="▶  Run Single Inference",
                                         command=self._start_single)
        self.btn_run_single.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_run_batch = ttk.Button(sec3, text="▶▶  Run Batch Test",
                                        command=self._start_batch)
        self.btn_run_batch.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_render = ttk.Button(sec3, text="🎬  Render Video",
                                     command=self._start_render)
        self.btn_render.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop = ttk.Button(sec3, text="⏹ Stop", command=self._stop_batch, state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_clear_log = ttk.Button(sec3, text="🗑 Clear Log", command=self._clear_log)
        self.btn_clear_log.pack(side=tk.RIGHT)

        # Render options row
        sec3b = ttk.Frame(mf); sec3b.pack(fill=tk.X, pady=(0, 4))
        self.save_render_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sec3b, text="Simpan video hasil render (.mp4)",
                        variable=self.save_render_var).pack(side=tk.LEFT)
        self.playback_speed_var = tk.StringVar(value="1.0x")
        ttk.Label(sec3b, text="  Kecepatan Playback:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Combobox(sec3b, textvariable=self.playback_speed_var,
                     values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"], width=6,
                     state="readonly").pack(side=tk.LEFT, padx=4)

        # ── Section 4: Log ──
        sec4 = ttk.LabelFrame(mf, text="3 · Log Pemrosesan", padding=8)
        sec4.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.log_text = tk.Text(sec4, height=10, wrap=tk.WORD,
                                font=("Consolas", 9), bg="#1e1e2e", fg="#cdd6f4",
                                insertbackground="#cdd6f4", selectbackground="#45475a")
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(sec4, command=self.log_text.yview)
        sb.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.config(yscrollcommand=sb.set)

        # ── Section 5: Single-file results ──
        sec5 = ttk.LabelFrame(mf, text="4 · Hasil Prediksi (Single)", padding=12)
        sec5.pack(fill=tk.X, pady=(0, 8))

        self.single_result_lbl = ttk.Label(sec5, text="Menunggu input…", style="Result.TLabel")
        self.single_result_lbl.pack(anchor="w", pady=(0, 6))

        self.gt_result_lbl = ttk.Label(sec5, text="", font=("Segoe UI", 10))
        self.gt_result_lbl.pack(anchor="w", pady=(0, 8))

        result_grid = ttk.Frame(sec5); result_grid.pack(fill=tk.X)
        self.class_bars = {}
        self.class_prob_lbls = {}
        for i, cls in enumerate(CLASSES):
            ttk.Label(result_grid, text=f"{cls}:", font=("Consolas", 11, "bold")).grid(
                row=i, column=0, sticky="w", pady=3)
            bar = ttk.Progressbar(result_grid, length=340, mode="determinate",
                                  style=f"{cls}.Horizontal.TProgressbar")
            bar.grid(row=i, column=1, padx=10, pady=3)
            self.class_bars[cls] = bar
            lbl = ttk.Label(result_grid, text="  0.00%", font=("Consolas", 11, "bold"), width=9)
            lbl.grid(row=i, column=2, sticky="e", pady=3)
            self.class_prob_lbls[cls] = lbl

        # ── Section 6: Batch results ──
        sec6 = ttk.LabelFrame(mf, text="5 · Hasil Batch Test", padding=12)
        sec6.pack(fill=tk.X, pady=(0, 8))

        # Summary labels
        self.batch_summary_lbl = ttk.Label(sec6, text="Belum ada batch test.", font=("Segoe UI", 10))
        self.batch_summary_lbl.pack(anchor="w", pady=(0, 6))

        # Treeview table
        cols = ("File", "Ground Truth", "Predicted", "Confidence", "Status")
        self.tree = ttk.Treeview(sec6, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("File",         width=220, minwidth=120)
        self.tree.column("Ground Truth", width=100, anchor="center")
        self.tree.column("Predicted",    width=100, anchor="center")
        self.tree.column("Confidence",   width=90,  anchor="center")
        self.tree.column("Status",       width=70,  anchor="center")
        self.tree.pack(fill=tk.X, pady=(0, 8))

        tree_sb = ttk.Scrollbar(sec6, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_sb.set)

        # Confusion-matrix canvas
        self.cm_canvas = tk.Canvas(sec6, width=360, height=300, bg="white", highlightthickness=1,
                                   highlightbackground="#bdc3c7")
        self.cm_canvas.pack(pady=(4, 8))

        # ── Section 7: Export ──
        sec7 = ttk.Frame(mf); sec7.pack(fill=tk.X, pady=(0, 12))
        self.btn_export_csv = ttk.Button(sec7, text="📄  Export Batch → CSV", command=self._export_csv)
        self.btn_export_csv.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_export_json = ttk.Button(sec7, text="📄  Export Batch → JSON", command=self._export_json)
        self.btn_export_json.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_result_dir_change(self, _=None):
        self.current_result_dir = self.result_dir_var.get()
        self.current_scenario = None   # force reload
        self.model = None
        self._log(f"Result folder diubah ke: {self.current_result_dir}")

    def _on_scenario_change(self, _=None):
        sc = int(self.scenario_var.get())
        self.scenario_desc_lbl.config(text=SCENARIOS[sc]["name"])
        self.current_scenario = None
        self.model = None
        self._log(f"Skenario diubah ke {sc}.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_initial_model(self):
        try:
            self._load_model(int(self.scenario_var.get()))
        except Exception as e:
            self._log(f"⚠ Gagal memuat model awal: {e}")

    def _load_model(self, scenario_id: int):
        if self.current_scenario == scenario_id and self.model is not None:
            return
        self._log(f"Memuat bobot model Skenario {scenario_id} dari '{self.current_result_dir}'…")
        params = SCENARIOS[scenario_id]
        model = AttentionBiLSTM(
            input_size=144,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            num_classes=4,
            dropout_rate=params["dropout"],
        )
        wpath = os.path.join(ROOT_DIR, self.current_result_dir,
                             f"scenario_{scenario_id}", "models", "best_model.pth")
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"File bobot tidak ditemukan: {wpath}")
        state = torch.load(wpath, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model
        self.current_scenario = scenario_id
        self._log(f"✓ Model Skenario {scenario_id} berhasil dimuat.")

    # ------------------------------------------------------------------
    # File / folder browser
    # ------------------------------------------------------------------
    def _browse_file(self):
        fp = filedialog.askopenfilename(
            title="Pilih File Input",
            initialdir=os.path.join(ROOT_DIR, "Dataset"),
            filetypes=[("Supported", "*.npy *.mp4 *.avi *.mov *.mkv"),
                       ("Numpy", "*.npy"), ("Video", "*.mp4 *.avi *.mov *.mkv"),
                       ("All", "*.*")],
        )
        if fp:
            self.file_path_var.set(fp)
            self._log(f"File dipilih: {fp}")

    def _browse_folder(self):
        fd = filedialog.askdirectory(
            title="Pilih Folder untuk Batch Test",
            initialdir=os.path.join(ROOT_DIR, "Dataset"),
        )
        if fd:
            self.folder_path_var.set(fd)
            self._log(f"Folder dipilih: {fd}")

    # ------------------------------------------------------------------
    # Ground-truth helper
    # ------------------------------------------------------------------
    def _resolve_gt(self, file_path: str) -> str | None:
        """Return ground-truth class name or None."""
        gt = self.gt_var.get()
        if gt != "Auto-detect":
            return gt
        # Auto-detect from parent directory name
        parent = os.path.basename(os.path.dirname(file_path))
        if parent in CLASSES:
            return parent
        return None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, file_path: str) -> np.ndarray:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_NUMPY:
            self._log(f"  Memuat numpy: {os.path.basename(file_path)}")
            seq = np.load(file_path)
            if seq.shape[0] != 30:
                seq = pad_or_truncate_sequence(seq)
            return seq
        elif ext in SUPPORTED_VIDEO:
            self._log(f"  Membaca video: {os.path.basename(file_path)}")
            frames = get_video_frames(file_path)
            self._log("  Menjalankan MediaPipe Pose Landmarker…")
            raw = extract_keypoints(frames)              # (N, 132)
            self._log(f"  Berhasil mengekstrak {raw.shape[0]} frame.")
            self._log("  Menghitung fitur rekayasa (kinematika & stance)…")
            feat = compute_engineered_features(raw)       # (N, 144)
            seq = pad_or_truncate_sequence(feat)
            return seq
        else:
            raise ValueError(f"Format tidak didukung: {ext}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _infer(self, sequence: np.ndarray):
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return CLASSES[idx], probs[idx] * 100, probs

    # ------------------------------------------------------------------
    # Sliding Window Classification + Temporal NMS
    # ------------------------------------------------------------------
    def _sliding_window_classify(self, raw_keypoints: np.ndarray) -> list:
        """
        Perform sliding window classification with window_size=30, stride=1.
        Each window [i, i+30) is fed to the model and the prediction is
        assigned to the LAST frame of that window (i+29).  Leading frames
        that cannot form a full window inherit the first window's prediction.

        Returns
        -------
        list of (pred_class, confidence, probs)  – one entry per frame.
        """
        window_size = 30  # MAX_TIMESTEPS
        N = raw_keypoints.shape[0]

        if N <= window_size:
            padded_seq = pad_or_truncate_sequence(raw_keypoints)
            pred_class, confidence, probs = self._infer(padded_seq)
            return [(pred_class, confidence, probs)] * N

        # Run inference for every window position (stride = 1)
        window_preds = []
        total_windows = N - window_size + 1
        for i in range(total_windows):
            window_seq = raw_keypoints[i:i + window_size]
            pred_class, confidence, probs = self._infer(window_seq)
            window_preds.append((pred_class, confidence, probs))

            if (i + 1) % 50 == 0 or i == total_windows - 1:
                self._log(f"    Window {i + 1}/{total_windows} selesai.")

        # Assign each window's prediction to the last frame of that window
        frame_preds = [None] * N
        for i, pred in enumerate(window_preds):
            frame_preds[i + window_size - 1] = pred

        # Fill leading frames (0 .. window_size-2) with the first window's prediction
        first_pred = window_preds[0]
        for i in range(window_size - 1):
            frame_preds[i] = first_pred

        return frame_preds

    def _apply_temporal_nms(self, frame_predictions: list,
                            min_segment_frames: int = 10) -> list:
        """
        Temporal Non-Maximum Suppression.
        Short 'burst' segments (fewer than *min_segment_frames* consecutive
        frames of the same class) are replaced by the prediction of the
        longer adjacent segment.  This prevents rapid flickering.
        """
        if not frame_predictions or len(frame_predictions) < 2:
            return frame_predictions

        N = len(frame_predictions)
        result = list(frame_predictions)

        # Iterate until stable (max 5 passes)
        for _ in range(5):
            # Identify contiguous segments of same-class predictions
            segments = []
            seg_start = 0
            for i in range(1, N):
                if result[i][0] != result[seg_start][0]:
                    segments.append((seg_start, i - 1, result[seg_start][0]))
                    seg_start = i
            segments.append((seg_start, N - 1, result[seg_start][0]))

            if len(segments) <= 1:
                break

            any_change = False
            for idx, (start, end, cls) in enumerate(segments):
                duration = end - start + 1
                if duration < min_segment_frames:
                    # Pick the longer adjacent segment as donor
                    prev_len = (segments[idx - 1][1] - segments[idx - 1][0] + 1) if idx > 0 else 0
                    next_len = (segments[idx + 1][1] - segments[idx + 1][0] + 1) if idx < len(segments) - 1 else 0

                    if prev_len >= next_len and prev_len > 0:
                        donor_frame = segments[idx - 1][1]    # last frame of prev segment
                    elif next_len > 0:
                        donor_frame = segments[idx + 1][0]    # first frame of next segment
                    else:
                        continue

                    for f in range(start, end + 1):
                        result[f] = result[donor_frame]
                    any_change = True

            if not any_change:
                break

        return result

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------
    def _start_single(self):
        fp = self.file_path_var.get().strip()
        if not fp:
            messagebox.showwarning("Input Kosong", "Silakan pilih file terlebih dahulu.")
            return
        if not os.path.isfile(fp):
            messagebox.showerror("File Tidak Ditemukan", f"File tidak ada:\n{fp}")
            return
        self._set_running(True)
        threading.Thread(target=self._single_worker, args=(fp,), daemon=True).start()

    def _single_worker(self, fp: str):
        try:
            self._load_model(int(self.scenario_var.get()))
            seq = self._preprocess(fp)
            pred, conf, probs = self._infer(seq)
            gt = self._resolve_gt(fp)

            self._log(f"  ➜ Prediksi: {pred} ({conf:.2f}%)")
            if gt:
                match = "✓ BENAR" if gt == pred else "✗ SALAH"
                self._log(f"  ➜ Ground Truth: {gt}  |  {match}")

            self.root.after(0, self._update_single_ui, pred, conf, probs, gt)
        except Exception as e:
            self._log(f"❌ Error: {e}")
            self.root.after(0, lambda: self.single_result_lbl.config(
                text="Error", foreground="red"))
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _update_single_ui(self, pred, conf, probs, gt):
        self.single_result_lbl.config(text=f"Prediksi: {pred}  ({conf:.2f}%)",
                                      foreground=CLASS_COLORS.get(pred, "#16213e"))
        if gt:
            match = "✓ BENAR" if gt == pred else "✗ SALAH"
            color = "#27ae60" if gt == pred else "#e74c3c"
            self.gt_result_lbl.config(text=f"Ground Truth: {gt}   →   {match}", foreground=color)
        else:
            self.gt_result_lbl.config(text="Ground Truth: tidak tersedia", foreground="#888")

        for i, cls in enumerate(CLASSES):
            v = probs[i] * 100
            self.class_bars[cls]["value"] = v
            self.class_prob_lbls[cls].config(text=f"{v:6.2f}%")

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    def _start_batch(self):
        folder = self.folder_path_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Folder Kosong", "Pilih folder yang valid untuk batch test.")
            return
        self._set_running(True)
        threading.Thread(target=self._batch_worker, args=(folder,), daemon=True).start()

    def _stop_batch(self):
        self._is_running = False
        self._log("⏹ Stop diminta oleh pengguna.")

    def _batch_worker(self, folder: str):
        try:
            self._load_model(int(self.scenario_var.get()))

            # Collect files recursively
            files = []
            for root_d, _, fnames in os.walk(folder):
                for fn in sorted(fnames):
                    if os.path.splitext(fn)[1].lower() in SUPPORTED_ALL:
                        files.append(os.path.join(root_d, fn))

            if not files:
                self._log("⚠ Tidak ada file yang didukung ditemukan di folder tersebut.")
                return

            self._log(f"📂 Batch test dimulai – {len(files)} file ditemukan.")
            self.root.after(0, self._clear_tree)
            self.batch_results.clear()

            correct = 0
            total_with_gt = 0

            for idx, fp in enumerate(files):
                if not self._is_running:
                    self._log("⏹ Batch test dihentikan.")
                    break
                self._log(f"[{idx+1}/{len(files)}] {os.path.basename(fp)}")
                try:
                    seq = self._preprocess(fp)
                    pred, conf, probs = self._infer(seq)
                    gt = self._resolve_gt(fp)
                    status = ""
                    if gt:
                        total_with_gt += 1
                        if gt == pred:
                            correct += 1
                            status = "✓"
                        else:
                            status = "✗"
                    entry = {"file": os.path.basename(fp), "path": fp,
                             "ground_truth": gt or "-", "predicted": pred,
                             "confidence": round(conf, 2),
                             "probs": {c: round(float(probs[i])*100, 2) for i, c in enumerate(CLASSES)},
                             "status": status}
                    self.batch_results.append(entry)
                    self.root.after(0, self._add_tree_row, entry)
                except Exception as e:
                    self._log(f"  ⚠ Gagal: {e}")
                    entry = {"file": os.path.basename(fp), "path": fp,
                             "ground_truth": "-", "predicted": "ERROR",
                             "confidence": 0, "probs": {}, "status": "ERR"}
                    self.batch_results.append(entry)
                    self.root.after(0, self._add_tree_row, entry)

            # Summary
            acc_str = f"{correct}/{total_with_gt} ({100*correct/total_with_gt:.1f}%)" if total_with_gt else "N/A (tanpa ground truth)"
            summary = f"Selesai: {len(self.batch_results)} file diproses  |  Akurasi: {acc_str}"
            self._log(f"📊 {summary}")
            self.root.after(0, lambda: self.batch_summary_lbl.config(text=summary))
            self.root.after(0, self._draw_confusion_matrix)

        except Exception as e:
            self._log(f"❌ Batch error: {e}")
        finally:
            self.root.after(0, lambda: self._set_running(False))

    # ------------------------------------------------------------------
    # Treeview helpers
    # ------------------------------------------------------------------
    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _add_tree_row(self, entry: dict):
        self.tree.insert("", tk.END, values=(
            entry["file"], entry["ground_truth"],
            entry["predicted"], f"{entry['confidence']:.1f}%", entry["status"]))

    # ------------------------------------------------------------------
    # Confusion matrix on Canvas
    # ------------------------------------------------------------------
    def _draw_confusion_matrix(self):
        c = self.cm_canvas
        c.delete("all")

        # Build matrix
        n = len(CLASSES)
        mat = [[0]*n for _ in range(n)]
        has_gt = False
        for r in self.batch_results:
            if r["ground_truth"] in CLASSES and r["predicted"] in CLASSES:
                has_gt = True
                ti = CLASSES.index(r["ground_truth"])
                pi = CLASSES.index(r["predicted"])
                mat[ti][pi] += 1

        if not has_gt:
            c.create_text(180, 150, text="Confusion matrix tidak tersedia\n(tidak ada ground truth)",
                          font=("Segoe UI", 10), fill="#888")
            return

        cw, ch = 360, 300
        margin_l, margin_t = 90, 50
        cell = min((cw - margin_l - 20) // n, (ch - margin_t - 20) // n)

        c.create_text(margin_l + (cell * n) // 2, 14, text="Confusion Matrix",
                      font=("Segoe UI", 11, "bold"), fill="#1a1a2e")
        c.create_text(margin_l + (cell * n) // 2, 32, text="Predicted →",
                      font=("Segoe UI", 8), fill="#555")
        c.create_text(14, margin_t + (cell * n) // 2, text="Actual ↓",
                      font=("Segoe UI", 8), fill="#555", angle=90)

        max_val = max(max(row) for row in mat) or 1
        for i in range(n):
            # Row label
            c.create_text(margin_l - 6, margin_t + i * cell + cell // 2,
                          text=CLASSES[i], anchor="e", font=("Consolas", 9, "bold"))
            # Col label
            c.create_text(margin_l + i * cell + cell // 2, margin_t - 6,
                          text=CLASSES[i], anchor="s", font=("Consolas", 9, "bold"))
            for j in range(n):
                x0 = margin_l + j * cell
                y0 = margin_t + i * cell
                v = mat[i][j]
                # Heat colour
                intensity = v / max_val
                if i == j:
                    r_c = int(39 + (46 - 39) * (1 - intensity))
                    g_c = int(174 + (204 - 174) * (1 - intensity))
                    b_c = int(96 + (153 - 96) * (1 - intensity))
                else:
                    r_c = int(231 + (245 - 231) * (1 - intensity))
                    g_c = int(76 + (245 - 76) * (1 - intensity))
                    b_c = int(60 + (245 - 60) * (1 - intensity))
                fill = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
                c.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=fill, outline="#bdc3c7")
                c.create_text(x0 + cell // 2, y0 + cell // 2, text=str(v),
                              font=("Consolas", 11, "bold"),
                              fill="white" if intensity > 0.5 else "#333")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self.batch_results:
            messagebox.showinfo("Kosong", "Belum ada data batch untuk di-export.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                          filetypes=[("CSV", "*.csv")],
                                          initialfile="batch_results.csv")
        if not fp:
            return
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "ground_truth", "predicted",
                                               "confidence", "status",
                                               *[f"prob_{c}" for c in CLASSES]])
            w.writeheader()
            for r in self.batch_results:
                row = {k: r[k] for k in ["file", "ground_truth", "predicted", "confidence", "status"]}
                for c in CLASSES:
                    row[f"prob_{c}"] = r.get("probs", {}).get(c, "")
                w.writerow(row)
        self._log(f"📄 CSV disimpan: {fp}")
        messagebox.showinfo("Export CSV", f"Berhasil disimpan:\n{fp}")

    def _export_json(self):
        if not self.batch_results:
            messagebox.showinfo("Kosong", "Belum ada data batch untuk di-export.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".json",
                                          filetypes=[("JSON", "*.json")],
                                          initialfile="batch_results.json")
        if not fp:
            return
        out = {"exported_at": datetime.datetime.now().isoformat(),
               "scenario": int(self.scenario_var.get()),
               "result_dir": self.current_result_dir,
               "total_files": len(self.batch_results),
               "results": self.batch_results}
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        self._log(f"📄 JSON disimpan: {fp}")
        messagebox.showinfo("Export JSON", f"Berhasil disimpan:\n{fp}")

    # ------------------------------------------------------------------
    # UI state helper
    # ------------------------------------------------------------------
    def _set_running(self, running: bool):
        self._is_running = running
        state_off = "disabled" if running else "normal"
        state_stop = "normal" if running else "disabled"
        for btn in (self.btn_run_single, self.btn_run_batch,
                    self.btn_browse_file, self.btn_browse_folder,
                    self.btn_render):
            btn.config(state=state_off)
        self.btn_stop.config(state=state_stop)

    # ------------------------------------------------------------------
    # OpenCV Video Rendering
    # ------------------------------------------------------------------
    def _start_render(self):
        fp = self.file_path_var.get().strip()
        if not fp:
            messagebox.showwarning("Input Kosong", "Pilih file video terlebih dahulu.")
            return
        ext = os.path.splitext(fp)[1].lower()
        if ext not in SUPPORTED_VIDEO:
            messagebox.showwarning("Format Salah",
                                   "Fitur Render Video hanya mendukung file video\n"
                                   f"(.mp4, .avi, .mov, .mkv).\n\nFile: {os.path.basename(fp)}")
            return
        if not os.path.isfile(fp):
            messagebox.showerror("File Tidak Ditemukan", f"File tidak ada:\n{fp}")
            return
        self._set_running(True)
        threading.Thread(target=self._render_worker, args=(fp,), daemon=True).start()

    def _render_worker(self, video_path: str):
        """Process video frame-by-frame: detect pose, classify, render overlay."""
        import mediapipe as mp

        try:
            self._load_model(int(self.scenario_var.get()))

            # Parse playback speed
            speed_str = self.playback_speed_var.get().replace('x', '')
            try:
                playback_speed = float(speed_str)
            except ValueError:
                playback_speed = 1.0
            if playback_speed <= 0:
                playback_speed = 1.0

            # ── Phase 1: Extract all frames + keypoints ──
            self._log("🎬 [Render] Fase 1: Membaca video dan mengekstrak pose…")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Tidak dapat membuka video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._log(f"  Video: {width}x{height} @ {fps:.1f} FPS, ~{total_frames} frame")

            # Read all frames (we need them for both MediaPipe extraction + rendering)
            bgr_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                bgr_frames.append(frame)
            cap.release()
            self._log(f"  Berhasil membaca {len(bgr_frames)} frame.")

            # Extract keypoints using the existing pipeline
            self._log("  Menjalankan MediaPipe Pose Landmarker…")
            rgb_gen = (cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames)
            raw_keypoints = extract_keypoints(rgb_gen)  # shape: (N, 132)
            self._log(f"  Keypoints diekstrak: {raw_keypoints.shape}")
            self._log("  Menghitung fitur rekayasa (kinematika & stance)…")
            raw_keypoints = compute_engineered_features(raw_keypoints)  # (N, 144)
            self._log(f"  Fitur total: {raw_keypoints.shape}")

            # Also extract RAW (unnormalised) landmarks for visualisation overlay
            # We do a second lightweight pass with MediaPipe to get pixel coordinates
            self._log("  Mengekstrak landmark mentah untuk overlay visual…")
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            mp_model_path = os.path.join(ROOT_DIR, 'src', 'preprocessing', 'pose_landmarker.task')

            raw_landmarks = []   # list of list-of-(x_px, y_px, vis) per frame
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=mp_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            with PoseLandmarker.create_from_options(options) as landmarker:
                ts_ms = 0
                for bgr_frame in bgr_frames:
                    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, ts_ms)
                    ts_ms += int(1000 / fps)

                    frame_lms = []
                    if result.pose_landmarks:
                        for lm in result.pose_landmarks[0]:
                            px = int(lm.x * width)
                            py = int(lm.y * height)
                            vis = lm.visibility if lm.visibility is not None else 0.0
                            frame_lms.append((px, py, vis))
                    raw_landmarks.append(frame_lms)

            # ── Phase 2: Sliding Window Classification (stride=1) ──
            self._log("🎬 [Render] Fase 2: Sliding window klasifikasi (stride=1)…")
            frame_predictions = self._sliding_window_classify(raw_keypoints)
            self._log(f"  Raw predictions: {len(frame_predictions)} frame")

            # Apply temporal NMS to remove short flickering segments
            self._log("  Menerapkan Temporal Non-Maximum Suppression…")
            frame_predictions = self._apply_temporal_nms(frame_predictions,
                                                         min_segment_frames=10)
            self._log("  ✓ NMS selesai.")

            gt = self._resolve_gt(video_path)

            # Compute overall majority-vote prediction for the summary UI
            class_counts = Counter(p[0] for p in frame_predictions)
            majority_class = class_counts.most_common(1)[0][0]
            majority_confs = [p[1] for p in frame_predictions if p[0] == majority_class]
            majority_conf = sum(majority_confs) / len(majority_confs)
            majority_probs = np.mean(
                [p[2] for p in frame_predictions if p[0] == majority_class], axis=0)
            self._log(f"  Prediksi mayoritas: {majority_class} ({majority_conf:.1f}%)")

            # Update the single-result UI with the majority vote
            self.root.after(0, self._update_single_ui,
                            majority_class, majority_conf, majority_probs, gt)

            # ── Phase 3: Render and play back ──
            self._log("🎬 [Render] Fase 3: Merender video dengan overlay…")
            self._log("  Tekan 'Q' pada jendela OpenCV untuk menutup.")
            self._log("  Tekan 'S' untuk menyimpan screenshot frame saat ini.")
            self._log("  Tekan SPASI untuk pause/resume.")

            # Setup video writer if user wants to save
            writer = None
            save_path = None
            if self.save_render_var.get():
                base = os.path.splitext(os.path.basename(video_path))[0]
                save_path = os.path.join(ROOT_DIR, 'results',
                                         f'rendered_{base}_{pred_class}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                self._log(f"  Video akan disimpan ke: {save_path}")

            frame_delay_base = 1.0 / fps
            paused = False
            window_name = f"Render: {os.path.basename(video_path)}"

            for fidx, bgr_frame in enumerate(bgr_frames):
                if not self._is_running:
                    self._log("⏹ Render dihentikan.")
                    break

                # ── Per-frame prediction from sliding window ──
                pred_class, confidence, probs = frame_predictions[fidx]
                pred_color = CLASS_COLORS_BGR.get(pred_class, (255, 255, 255))

                canvas = bgr_frame.copy()

                # Draw skeleton + landmarks
                if fidx < len(raw_landmarks) and raw_landmarks[fidx]:
                    lms = raw_landmarks[fidx]
                    # Draw connections
                    for (a, b) in POSE_CONNECTIONS:
                        if a < len(lms) and b < len(lms):
                            if lms[a][2] > 0.3 and lms[b][2] > 0.3:
                                cv2.line(canvas, (lms[a][0], lms[a][1]),
                                         (lms[b][0], lms[b][1]),
                                         (0, 255, 255), 2, cv2.LINE_AA)
                    # Draw joints
                    for (px, py, vis) in lms:
                        if vis > 0.3:
                            cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)
                            cv2.circle(canvas, (px, py), 5, (0, 180, 0), 1, cv2.LINE_AA)

                # ── Overlay: classification result panel ──
                panel_h = 140
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (width, panel_h), (20, 20, 30), -1)
                cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

                # Title text
                cv2.putText(canvas, "Boxing Punch Classifier", (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

                # Predicted class
                cv2.putText(canvas, f"{pred_class}", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3, cv2.LINE_AA)
                cv2.putText(canvas, f"{confidence:.1f}%", (15 + len(pred_class) * 30 + 10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                # Mini probability bars
                bar_x = 15
                bar_y = 80
                bar_w = 180
                bar_h = 12
                for ci, cls in enumerate(CLASSES):
                    p = probs[ci]
                    color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
                    y = bar_y + ci * (bar_h + 5)
                    # Label
                    cv2.putText(canvas, f"{cls[:3]}", (bar_x, y + bar_h - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
                    # Background bar
                    bx = bar_x + 35
                    cv2.rectangle(canvas, (bx, y), (bx + bar_w, y + bar_h), (60, 60, 70), -1)
                    # Filled bar
                    fill_w = int(bar_w * p)
                    if fill_w > 0:
                        cv2.rectangle(canvas, (bx, y), (bx + fill_w, y + bar_h), color, -1)
                    # Percentage
                    cv2.putText(canvas, f"{p*100:.0f}%", (bx + bar_w + 6, y + bar_h - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

                # Ground truth badge (top-right)
                if gt:
                    gt_text = f"GT: {gt}"
                    match_text = "BENAR" if gt == pred_class else "SALAH"
                    badge_color = (96, 174, 39) if gt == pred_class else (60, 76, 231)
                    cv2.putText(canvas, f"{gt_text} | {match_text}",
                                (width - 250, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, badge_color, 2, cv2.LINE_AA)

                # Frame counter (bottom-left)
                cv2.putText(canvas, f"Frame {fidx+1}/{len(bgr_frames)}",
                            (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (180, 180, 180), 1, cv2.LINE_AA)

                if writer:
                    writer.write(canvas)

                cv2.imshow(window_name, canvas)

                # Handle keyboard
                wait_ms = max(1, int(frame_delay_base / playback_speed * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q') or key == 27:      # Q or ESC
                    self._log("  Jendela ditutup oleh pengguna.")
                    break
                elif key == ord('s'):
                    ss_path = os.path.join(ROOT_DIR, 'results',
                                           f'screenshot_{os.path.splitext(os.path.basename(video_path))[0]}_f{fidx}.png')
                    cv2.imwrite(ss_path, canvas)
                    self._log(f"  📸 Screenshot disimpan: {ss_path}")
                elif key == ord(' '):
                    paused = not paused
                    self._log(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
                    while paused and self._is_running:
                        pk = cv2.waitKey(100) & 0xFF
                        if pk == ord(' '):
                            paused = False
                            self._log("  ▶ Resumed")
                        elif pk == ord('q') or pk == 27:
                            paused = False
                            self._is_running = False

                # Check if window was closed via X button
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            cv2.destroyAllWindows()
            if writer:
                writer.release()
                self._log(f"✓ Video tersimpan: {save_path}")
            self._log("🎬 [Render] Selesai.")

        except Exception as e:
            self._log(f"❌ Render error: {e}")
            cv2.destroyAllWindows()
        finally:
            self.root.after(0, lambda: self._set_running(False))


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = PunchClassifierGUI(root)
    root.mainloop()
