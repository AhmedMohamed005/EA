# gui.py
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore, QtGui


try:
    from .hybrid import HybridOptimizer, set_global_seed
except ImportError:
    # Stubs so the file can be opened/previewed standalone
    def set_global_seed(s): np.random.seed(s)
    class HybridOptimizer:
        def __init__(self, dataset_path): pass
        class ga:
            population_size = 20
        def run_generation(self, pop):
            n = len(pop)
            fits = np.random.uniform(100, 500, n)
            return pop, fits
        def get_sample_route(self, layout):
            route = list(range(len(layout)))
            locs = [(i//10, i%10) for i in range(len(layout))]
            return route, locs


# ============================================================
#  COLOUR / STYLE CONSTANTS
# ============================================================
DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
CARD_BG      = "#1c2333"
BORDER       = "#30363d"
ACCENT_TEAL  = "#39d1a8"
ACCENT_AMBER = "#f0a429"
ACCENT_RED   = "#e24b4a"
ACCENT_BLUE  = "#378add"
TEXT_PRI     = "#e6edf3"
TEXT_SEC     = "#8b949e"
TEXT_MUT     = "#484f58"

BTN_GREEN    = "#238636"
BTN_GREEN_HV = "#2ea043"
BTN_RED_HV   = "#da3633"

MONO_FONT    = "JetBrains Mono, Consolas, Courier New, monospace"

# Matplotlib theme
plt.rcParams.update({
    "figure.facecolor":    DARK_BG,
    "axes.facecolor":      PANEL_BG,
    "axes.edgecolor":      BORDER,
    "axes.labelcolor":     TEXT_SEC,
    "axes.titlecolor":     TEXT_PRI,
    "xtick.color":         TEXT_MUT,
    "ytick.color":         TEXT_MUT,
    "grid.color":          BORDER,
    "grid.alpha":          0.6,
    "text.color":          TEXT_PRI,
    "lines.linewidth":     2,
    "font.family":         "monospace",
})


# ============================================================
#  WORKER THREAD  (unchanged logic, improved error surfacing)
# ============================================================
class WorkerThread(QtCore.QThread):
    update_gui    = QtCore.pyqtSignal(list, float, int, int, list, list)
    finished_sim  = QtCore.pyqtSignal(float)   # sends total elapsed secs
    error_sim     = QtCore.pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs  = kwargs
        self.running = True

    def run(self):
        t0 = time.time()
        try:
            base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "dataset", "synthetic_warehouse_orders.csv")

            set_global_seed(self.kwargs["seed"])
            optimizer = HybridOptimizer(dataset_path=dataset_path)
            optimizer.ga.population_size = self.kwargs["pop"]

            gens       = self.kwargs["gens"]
            population = optimizer.ga.initialize_population() if hasattr(optimizer.ga, "initialize_population") \
                         else [list(np.random.permutation(100)) for _ in range(self.kwargs["pop"])]

            for g in range(gens):
                if not self.running:
                    break
                new_pop, fitnesses = optimizer.run_generation(population)
                population  = new_pop
                best_idx    = int(np.argmin(fitnesses))
                best_fit    = float(fitnesses[best_idx])
                best_layout = population[best_idx]
                best_route, locs = optimizer.get_sample_route(best_layout)
                self.update_gui.emit(best_layout, best_fit, g + 1, gens, best_route, locs)

            self.finished_sim.emit(time.time() - t0)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.error_sim.emit(str(exc))

    def stop(self):
        self.running = False


# ============================================================
#  STAT CARD WIDGET
# ============================================================
class StatCard(QtWidgets.QFrame):
    def __init__(self, label, unit="", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self.lbl = QtWidgets.QLabel(label.upper())
        self.lbl.setStyleSheet(f"color: {TEXT_MUT}; font-size: 10px; letter-spacing: 1.5px; border: none; background: transparent;")

        self.val = QtWidgets.QLabel("—")
        self.val.setStyleSheet(f"color: {TEXT_PRI}; font-size: 22px; font-weight: bold; font-family: {MONO_FONT}; border: none; background: transparent;")

        self.unit_lbl = QtWidgets.QLabel(unit)
        self.unit_lbl.setStyleSheet(f"color: {TEXT_MUT}; font-size: 11px; border: none; background: transparent;")

        layout.addWidget(self.lbl)
        layout.addWidget(self.val)
        layout.addWidget(self.unit_lbl)

    def update_value(self, v, color=TEXT_PRI):
        if isinstance(v, float):
            text = f"{v:,.1f}"
        else:
            text = str(v)
        self.val.setText(text)
        self.val.setStyleSheet(
            f"color: {color}; font-size: 22px; font-weight: bold; font-family: {MONO_FONT}; border: none; background: transparent;"
        )


# ============================================================
#  MAIN GUI
# ============================================================
class WarehouseGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GA-ACO Warehouse Optimiser  //  AI420")
        self.setMinimumSize(1280, 800)
        self.resize(1360, 860)
        self._apply_global_style()

        self.generation_data  = []
        self.best_ever        = float("inf")
        self.worst_in_run     = float("-inf")
        self.worker           = None
        self.enable_animation = False
        self.animation_speed  = 0.008

        self._build_ui()

    # ----------------------------------------------------------
    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {DARK_BG};
                color: {TEXT_PRI};
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
            }}
            QLabel  {{ background: transparent; }}
            QLineEdit {{
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 6px;
                color: {TEXT_PRI};
                padding: 6px 10px;
                font-family: {MONO_FONT};
                font-size: 13px;
            }}
            QLineEdit:focus {{ border: 1px solid {ACCENT_TEAL}; }}
            QPushButton {{
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#startBtn {{
                background: {BTN_GREEN};
                color: #ffffff;
                border: 1px solid #3fb950;
            }}
            QPushButton#startBtn:hover  {{ background: {BTN_GREEN_HV}; }}
            QPushButton#startBtn:disabled {{ background: {TEXT_MUT}; color: {PANEL_BG}; border: none; }}
            QPushButton#stopBtn {{
                background: transparent;
                color: {ACCENT_RED};
                border: 1px solid {ACCENT_RED};
            }}
            QPushButton#stopBtn:hover {{ background: rgba(226,75,74,0.12); }}
            QPushButton#stopBtn:disabled {{ color: {TEXT_MUT}; border-color: {TEXT_MUT}; }}
            QPushButton#animBtn {{
                background: transparent;
                color: {ACCENT_BLUE};
                border: 1px solid {ACCENT_BLUE};
            }}
            QPushButton#animBtn:hover {{ background: rgba(55,138,221,0.12); }}
            QProgressBar {{
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 4px;
                height: 6px;
                text-align: center;
                font-size: 10px;
                color: transparent;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_TEAL}, stop:1 {ACCENT_BLUE});
                border-radius: 4px;
            }}
            QScrollArea {{ border: none; }}
        """)

    # ----------------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── TOP BAR ────────────────────────────────────────────
        topbar = QtWidgets.QFrame()
        topbar.setFixedHeight(48)
        topbar.setStyleSheet(f"background: {PANEL_BG}; border-bottom: 1px solid {BORDER};")
        tb_lay = QtWidgets.QHBoxLayout(topbar)
        tb_lay.setContentsMargins(20, 0, 20, 0)

        logo = QtWidgets.QLabel("⬡  WAREHOUSE OPTIMISER")
        logo.setStyleSheet(f"color: {ACCENT_TEAL}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        tb_lay.addWidget(logo)
        tb_lay.addStretch()

        badge = QtWidgets.QLabel("Hybrid GA-ACO  //  AI420 Spring 2026")
        badge.setStyleSheet(f"color: {TEXT_MUT}; font-size: 11px;")
        tb_lay.addWidget(badge)
        outer.addWidget(topbar)

        # ── BODY (sidebar + main) ───────────────────────────────
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        body.addWidget(self._build_sidebar(), 0)

        # divider
        div = QtWidgets.QFrame()
        div.setFixedWidth(1)
        div.setStyleSheet(f"background: {BORDER};")
        body.addWidget(div)

        body.addLayout(self._build_main_panel(), 1)

    # ----------------------------------------------------------
    def _build_sidebar(self):
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(270)
        sidebar.setStyleSheet(f"background: {PANEL_BG}; border: none;")
        lay = QtWidgets.QVBoxLayout(sidebar)
        lay.setContentsMargins(16, 20, 16, 16)
        lay.setSpacing(12)

        def section_header(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color: {TEXT_MUT}; font-size: 10px; letter-spacing: 1.5px; font-weight: bold;")
            return lbl

        def param_row(label, default, tip=""):
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px;")
            le  = QtWidgets.QLineEdit(str(default))
            if tip:
                le.setToolTip(tip)
            return lbl, le

        # ── PARAMETERS ─────────────────────────────────────────
        lay.addWidget(section_header("PARAMETERS"))

        self.inputs = {}
        params = [
            ("Generations",      50,   "Number of GA generations to run"),
            ("Population Size",  20,   "Number of layout candidates per generation"),
            ("Mutation Rate",    0.05, "Probability of mutation per gene"),
            ("Evaporation (ρ)",  0.10, "ACO pheromone evaporation rate"),
            ("Random Seed",      42,   "Fixed seed for reproducibility"),
        ]
        for name, default, tip in params:
            lbl, le = param_row(name, default, tip)
            self.inputs[name] = le
            lay.addWidget(lbl)
            lay.addWidget(le)
            if name != params[-1][0]:
                lay.addSpacing(2)

        lay.addSpacing(8)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        lay.addWidget(sep)
        lay.addSpacing(4)

        # ── CONTROLS ───────────────────────────────────────────
        lay.addWidget(section_header("CONTROLS"))

        self.start_btn = QtWidgets.QPushButton("▶  Run Optimisation")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_simulation)
        lay.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("■  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        lay.addWidget(self.stop_btn)

        self.anim_btn = QtWidgets.QPushButton("◉  Animation: OFF")
        self.anim_btn.setObjectName("animBtn")
        self.anim_btn.clicked.connect(self.toggle_animation)
        lay.addWidget(self.anim_btn)

        lay.addSpacing(8)
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setStyleSheet(f"color: {BORDER};")
        lay.addWidget(sep2)
        lay.addSpacing(4)

        # ── PROGRESS ───────────────────────────────────────────
        lay.addWidget(section_header("PROGRESS"))

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        lay.addWidget(self.progress)

        self.status_lbl = QtWidgets.QLabel("IDLE")
        self.status_lbl.setStyleSheet(f"color: {TEXT_MUT}; font-size: 11px; font-family: {MONO_FONT};")
        lay.addWidget(self.status_lbl)

        lay.addSpacing(8)
        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setStyleSheet(f"color: {BORDER};")
        lay.addWidget(sep3)
        lay.addSpacing(4)

        # ── LEGEND ─────────────────────────────────────────────
        lay.addWidget(section_header("LEGEND"))

        def legend_dot(color, text):
            row = QtWidgets.QHBoxLayout()
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            dot.setFixedWidth(20)
            txt = QtWidgets.QLabel(text)
            txt.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px;")
            txt.setWordWrap(True)
            row.addWidget(dot)
            row.addWidget(txt, 1)
            return row

        for color, text in [
            (ACCENT_BLUE,  "Shipping dock (origin)"),
            (ACCENT_RED,   "Hazardous material zone"),
            (ACCENT_TEAL,  "High-frequency item"),
            (ACCENT_AMBER, "ACO pheromone route"),
        ]:
            lay.addLayout(legend_dot(color, text))
            lay.addSpacing(2)

        lay.addStretch()

        # ── BOTTOM VERSION TAG ─────────────────────────────────
        ver = QtWidgets.QLabel("v2.0  •  AI420 EA Project")
        ver.setStyleSheet(f"color: {TEXT_MUT}; font-size: 10px;")
        ver.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(ver)

        return sidebar

    # ----------------------------------------------------------
    def _build_main_panel(self):
        main = QtWidgets.QVBoxLayout()
        main.setContentsMargins(20, 16, 20, 16)
        main.setSpacing(12)

        # ── STAT CARDS ─────────────────────────────────────────
        cards_row = QtWidgets.QHBoxLayout()
        cards_row.setSpacing(12)

        self.card_gen   = StatCard("Generation",  "/ —")
        self.card_best  = StatCard("Best Fitness", "dist")
        self.card_worst = StatCard("Worst Seen",   "dist")
        self.card_impr  = StatCard("Improvement",  "%")

        for card in (self.card_gen, self.card_best, self.card_worst, self.card_impr):
            cards_row.addWidget(card)
        main.addLayout(cards_row)

        # ── CHARTS ─────────────────────────────────────────────
        charts_row = QtWidgets.QHBoxLayout()
        charts_row.setSpacing(12)

        # Warehouse grid
        self.fig_grid = Figure(figsize=(5, 5), facecolor=DARK_BG)
        self.ax_grid  = self.fig_grid.add_subplot(111)
        self.canvas_grid = FigureCanvas(self.fig_grid)
        self.canvas_grid.setMinimumSize(400, 380)

        # Convergence chart
        self.fig_conv = Figure(figsize=(5, 4), facecolor=DARK_BG)
        self.ax_conv  = self.fig_conv.add_subplot(111)
        self.canvas_conv = FigureCanvas(self.fig_conv)
        self.canvas_conv.setMinimumSize(340, 380)

        charts_row.addWidget(self._wrap_canvas(self.canvas_grid,  "WAREHOUSE LAYOUT  +  ACO ROUTE"), 3)
        charts_row.addWidget(self._wrap_canvas(self.canvas_conv,  "CONVERGENCE CURVE"), 2)
        main.addLayout(charts_row, 1)

        self._init_grid_plot()
        self._init_conv_plot()

        return main

    def _wrap_canvas(self, canvas, title):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        hdr = QtWidgets.QLabel(title)
        hdr.setStyleSheet(
            f"color: {TEXT_MUT}; font-size: 10px; letter-spacing: 1.5px; font-weight: bold;"
        )
        lay.addWidget(hdr)
        lay.addWidget(canvas, 1)
        return frame

    # ----------------------------------------------------------
    def _init_grid_plot(self):
        ax = self.ax_grid
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Awaiting simulation…", color=TEXT_MUT, fontsize=10, pad=8)
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=False)
        ax.grid(color=BORDER, linewidth=0.5)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_grid.tight_layout(pad=1.2)
        self.canvas_grid.draw()

    def _init_conv_plot(self):
        ax = self.ax_conv
        ax.set_facecolor(PANEL_BG)
        ax.set_title("No data yet", color=TEXT_MUT, fontsize=10, pad=8)
        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel("Fitness (route dist + penalty)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.fig_conv.tight_layout(pad=1.2)
        self.canvas_conv.draw()

    # ----------------------------------------------------------
    def _draw_grid(self, layout, gen, max_gens):
        ax = self.ax_grid
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.set_title(f"Generation {gen} / {max_gens}", color=TEXT_PRI, fontsize=10, pad=8)
        ax.set_xticks(np.arange(-0.5, 10, 1))
        ax.set_yticks(np.arange(-0.5, 10, 1))
        ax.grid(color=BORDER, linewidth=0.5)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

        for i, item_idx in enumerate(layout):
            r, c = i // 10, i % 10
            # Default cell colour
            if r == 9:
                fc = "#3a1a1a"   # hazardous row – dark red tint
            elif r == 8:
                fc = "#4a4a1a"   # packing zone row – dark yellow tint
            elif (r, c) == (0, 0):
                fc = "#0c3058"   # dock – dark blue
            else:
                # Heat-map: higher item index = more popular
                intensity = item_idx / max(len(layout), 1)
                g_val = int(28 + intensity * 20)
                fc = f"#{g_val:02x}{g_val+4:02x}{g_val+8:02x}"

            rect = mpatches.FancyBboxPatch(
                (c - 0.47, r - 0.47), 0.94, 0.94,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor=BORDER, linewidth=0.4,
            )
            ax.add_patch(rect)

            # Highlight dock
            if (r, c) == (0, 0):
                ax.text(c, r, "DOCK", ha="center", va="center",
                        fontsize=5.5, color=ACCENT_BLUE, fontweight="bold")
            elif r == 9:
                ax.text(c, r, f"{item_idx}", ha="center", va="center",
                        fontsize=5.5, color="#e87070")
            else:
                ax.text(c, r, f"{item_idx}", ha="center", va="center",
                        fontsize=5.5, color=TEXT_MUT)

        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(9.5, -0.5)

    # ----------------------------------------------------------
    def _draw_route(self, route_xs, route_ys):
        ax = self.ax_grid
        # Gradient-like effect: draw segments with increasing alpha
        n = len(route_xs)
        for i in range(1, n):
            alpha = 0.35 + 0.65 * (i / n)
            ax.plot(
                [route_xs[i-1], route_xs[i]],
                [route_ys[i-1], route_ys[i]],
                color=ACCENT_AMBER, alpha=alpha, linewidth=1.8, solid_capstyle="round",
            )
        if route_xs:
            ax.scatter(route_xs[1:-1], route_ys[1:-1],
                       color=ACCENT_AMBER, s=14, zorder=5, alpha=0.7)
            ax.scatter([route_xs[0]], [route_ys[0]],
                       color=ACCENT_BLUE, s=40, zorder=6, marker="*")

    # ----------------------------------------------------------
    def _draw_conv(self):
        ax = self.ax_conv
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_title("Convergence", color=TEXT_PRI, fontsize=10, pad=8)
        ax.set_xlabel("Generation", fontsize=9, color=TEXT_SEC)
        ax.set_ylabel("Fitness", fontsize=9, color=TEXT_SEC)
        ax.grid(True, linestyle="--", alpha=0.3, color=BORDER)

        data = self.generation_data
        xs   = list(range(1, len(data) + 1))

        # Fill under curve
        ax.fill_between(xs, data, alpha=0.12, color=ACCENT_TEAL)
        ax.plot(xs, data, color=ACCENT_TEAL, linewidth=2, solid_capstyle="round")

        if len(data) >= 2:
            # Rolling min (best-so-far)
            best_so_far = [min(data[:i+1]) for i in range(len(data))]
            ax.plot(xs, best_so_far, color=ACCENT_BLUE, linewidth=1.2,
                    linestyle="--", alpha=0.7, label="Best so far")
            ax.legend(fontsize=8, framealpha=0.2,
                      labelcolor=TEXT_SEC, facecolor=CARD_BG,
                      edgecolor=BORDER)

        self.fig_conv.tight_layout(pad=1.2)
        self.canvas_conv.draw()

    # ----------------------------------------------------------
    def update_plots(self, layout, fitness, gen, max_gens, best_route, locs):
        self.generation_data.append(fitness)
        max_gens_total = int(self.inputs["Generations"].text())

        # Update progress
        pct = int(gen / max(max_gens, 1) * 100)
        self.progress.setValue(pct)
        self.status_lbl.setText(f"GEN {gen:>4} / {max_gens}  |  fitness {fitness:,.1f}")

        # Stat cards
        self.card_gen.update_value(f"{gen} / {max_gens}", ACCENT_TEAL)
        if fitness < self.best_ever:
            self.best_ever = fitness
        if fitness > self.worst_in_run:
            self.worst_in_run = fitness
        self.card_best.update_value(self.best_ever, ACCENT_TEAL)
        self.card_worst.update_value(self.worst_in_run, ACCENT_RED)

        if self.generation_data:
            first  = self.generation_data[0]
            improv = (first - self.best_ever) / max(first, 1e-9) * 100
            self.card_impr.update_value(round(improv, 1),
                                        ACCENT_TEAL if improv > 0 else TEXT_SEC)

        # Build route coordinates
        route_xs, route_ys = [], []
        if best_route and locs:
            for node in best_route:
                if node == 0:
                    route_xs.append(0); route_ys.append(0)
                else:
                    r, c = locs[node - 1]
                    route_xs.append(c); route_ys.append(r)

        if self.enable_animation and gen % 10 == 0 and route_xs:
            self._draw_grid(layout, gen, max_gens)
            self.canvas_grid.draw()
            QtWidgets.QApplication.processEvents()
            self._animate_route(route_xs, route_ys, layout, gen, max_gens)
        else:
            self._draw_grid(layout, gen, max_gens)
            if route_xs:
                self._draw_route(route_xs, route_ys)
            self.fig_grid.tight_layout(pad=1.2)
            self.canvas_grid.draw()

        self._draw_conv()

    # ----------------------------------------------------------
    def _animate_route(self, xs, ys, layout, gen, max_gens):
        px, py = [], []
        self._draw_grid(layout, gen, max_gens)
        for i in range(len(xs)):
            if not self.worker or not self.worker.running:
                break
            px.append(xs[i]); py.append(ys[i])
            self.ax_grid.plot(px, py, color=ACCENT_AMBER, linewidth=2,
                              marker="o", markersize=3, alpha=0.8)
            self.ax_grid.scatter([xs[i]], [ys[i]], color=ACCENT_BLUE, s=45, zorder=6)
            self.canvas_grid.draw()
            QtWidgets.QApplication.processEvents()
            time.sleep(self.animation_speed)

    # ----------------------------------------------------------
    def toggle_animation(self):
        self.enable_animation = not self.enable_animation
        state = "ON" if self.enable_animation else "OFF"
        self.anim_btn.setText(f"◉  Animation: {state}")

    # ----------------------------------------------------------
    def start_simulation(self):
        try:
            kwargs = {
                "gens": int(self.inputs["Generations"].text()),
                "pop":  int(self.inputs["Population Size"].text()),
                "seed": int(self.inputs["Random Seed"].text()),
            }
        except ValueError:
            self.status_lbl.setText("ERROR: inputs must be integers")
            return

        self.generation_data = []
        self.best_ever       = float("inf")
        self.worst_in_run    = float("-inf")
        self.progress.setValue(0)
        self.card_gen.update_value("—")
        self.card_best.update_value("—")
        self.card_worst.update_value("—")
        self.card_impr.update_value("—")
        self._init_grid_plot()
        self._init_conv_plot()

        # Update card subtitles
        total_lbl = f"/ {kwargs['gens']}"
        self.card_gen.unit_lbl.setText(total_lbl)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("RUNNING…")

        self.worker = WorkerThread(kwargs)
        self.worker.update_gui.connect(self.update_plots)
        self.worker.finished_sim.connect(self._sim_finished)
        self.worker.error_sim.connect(self._sim_error)
        self.worker.start()

    def stop_simulation(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self._sim_finished(0)
        self.status_lbl.setText("STOPPED")

    def _sim_finished(self, elapsed):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        if elapsed > 0:
            self.status_lbl.setText(f"DONE  |  elapsed {elapsed:.1f}s")

    def _sim_error(self, err):
        self._sim_finished(0)
        self.status_lbl.setText(f"ERROR: {err}")
        QtWidgets.QMessageBox.critical(self, "Simulation Error", err)


# ============================================================
#  ENTRY POINT
# ============================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Warehouse Optimiser")
    window = WarehouseGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()