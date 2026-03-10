import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont, filedialog
import cv2
import threading
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image, ImageTk
from reid import ReIDEngine, AnalyticsEngine # Ensure AnalyticsEngine is in reid.py
import os
from dotenv import load_dotenv

# load the .env file
load_dotenv()


# ==================== SETTINGS ====================
MODEL_PATH = 'model/osnet_x0_75_imagenet.pth'
VIDEO_1 = "./videos/cam6.2.mp4"
VIDEO_2 = "./videos/cam1.1.mp4"
MATCHES_DIR = "matches"
ANALYTICS_DIR = "analytics_output"
ZONE_ALERTS_DIR = "zone_alerts"
SUSPECT_FINDER_DIR = "suspect_results"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set your Groq API key here for suspect finder (optional)
# ==================================================

# ==================== THEME ====================
BG_DARK      = "#0a0e1a"
BG_PANEL     = "#0f1628"
BG_CARD      = "#141c30"
BG_SURFACE   = "#1a2240"
ACCENT_BLUE  = "#00a8ff"
ACCENT_CYAN  = "#00e5ff"
ACCENT_GREEN = "#00ff88"
ACCENT_RED   = "#ff4757"
ACCENT_AMBER = "#ffa502"
TEXT_PRIMARY = "#e8edf5"
TEXT_SECONDARY = "#7a8fa8"
TEXT_DIM     = "#3d4f6b"
BORDER       = "#1e2d4a"
BORDER_BRIGHT= "#2a3f5f"
# ===============================================

class SentinelVision(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SSF VISION  ·  AI Surveillance Platform")
        self.geometry("1400x900")
        self.minsize(1200, 750)
        self.configure(bg=BG_DARK)



        # Engines
        self.engine = ReIDEngine(MODEL_PATH)
        self.analytics_engine = AnalyticsEngine() # New Analytics Engine
        from reid import ZoneEngine # Import the new engine
        self.zone_engine = ZoneEngine()

        # Global State
        self.is_running = False
        self.current_frame_size = (640, 480)
        self.video_cap = None  # For zone monitoring video stream
        self.setup_frame = None  # First frame for zone drawing
        
        # Re-ID State
        self.selected_tid = None
        self.latest_detections = []
        self.match_queue = []
        self.match_results = []
        self.current_result_idx = 0
        self.search_complete = False
        self._active_items = []
        self._active_results_tab = "reid"

        # Suspect Finder state
        self.suspect_video_path = None
        self.suspect_is_running = False
        self.suspect_match_results = []

        # Anomaly Detection state
        self.anomaly_video_path = None
        self.anomaly_is_running = False
        self.anomaly_results = []
        self.anomaly_log_path = 'security_logs.csv'
        
        # Zoom state for results viewer
        self.result_zoom_level = 1.0  # 1.0 = 100% fit to display
        self.result_original_image = None  # Store original PIL image

        # Directory Setup
        os.makedirs(ANALYTICS_DIR, exist_ok=True)
        os.makedirs(ZONE_ALERTS_DIR, exist_ok=True)

        self._build_fonts()
        self._build_layout()
        self._start_clock()
        self.show_page("Home")

    def _build_fonts(self):
        self.font_title  = ("Courier New", 18, "bold")
        self.font_head   = ("Courier New", 13, "bold")
        self.font_sub     = ("Courier New", 10)
        self.font_mono   = ("Courier New", 9)
        self.font_huge   = ("Courier New", 32, "bold")
        self.font_btn    = ("Courier New", 10, "bold")

    def _start_clock(self):
        self._update_clock()

    def _update_clock(self):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        if hasattr(self, 'clock_lbl'):
            self.clock_lbl.config(text=now)
        self.after(1000, self._update_clock)

    def _build_layout(self):
        topbar = tk.Frame(self, bg=BG_PANEL, height=52)
        topbar.pack(side="top", fill="x")
        topbar.pack_propagate(False)

        logo_frame = tk.Frame(topbar, bg=BG_PANEL)
        logo_frame.pack(side="left", padx=20)
        tk.Label(logo_frame, text="⬡", font=("Courier New", 22, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side="left")
        tk.Label(logo_frame, text=" SSF", font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_PANEL).pack(side="left")
        tk.Label(logo_frame, text=" VISION", font=("Courier New", 16), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side="left")

        right = tk.Frame(topbar, bg=BG_PANEL)
        right.pack(side="right", padx=20)
        tk.Label(right, text="● LIVE", font=("Courier New", 9, "bold"), fg=ACCENT_GREEN, bg=BG_PANEL).pack(side="right", padx=10)
        self.clock_lbl = tk.Label(right, font=("Courier New", 10), fg=TEXT_SECONDARY, bg=BG_PANEL, text="")
        self.clock_lbl.pack(side="right", padx=10)

        tk.Frame(self, bg=ACCENT_CYAN, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(body, bg=BG_PANEL, width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        self.content = tk.Frame(body, bg=BG_DARK)
        self.content.pack(side="right", fill="both", expand=True)

        self.home_page      = tk.Frame(self.content, bg=BG_DARK)
        self.reid_page      = tk.Frame(self.content, bg=BG_DARK)
        self.analytics_page = tk.Frame(self.content, bg=BG_DARK)
        self.zone_page      = tk.Frame(self.content, bg=BG_DARK)
        self.results_page   = tk.Frame(self.content, bg=BG_DARK)
        self.suspect_page   = tk.Frame(self.content, bg=BG_DARK)
        self.anomaly_page   = tk.Frame(self.content, bg=BG_DARK)

        self._build_home_page()
        self._build_reid_page()
        self._build_analytics_page()
        self._build_zone_page()
        self._build_results_page()
        self._build_suspect_page()
        self._build_anomaly_page()

    def _build_sidebar(self):
        tk.Frame(self.sidebar, bg=BG_PANEL, height=20).pack()
        tk.Label(self.sidebar, text="NAVIGATION", font=("Courier New", 7, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(10, 4))

        nav_items = [
            ("Home",      "⌂", "Dashboard"),
            ("ReID",      "◉", "Person Re-ID"),
            ("Analytics", "📊", "People Analytics"),
            ("Zones",     "🚫", "Restricted Zones"),
            ("Suspect",   "🔍", "Suspect Finder"),
            ("Anomaly",   "⚠", "Anomaly Detection"),
            ("Results",   "▦", "Match Results"),
        ]
        self.nav_btns = {}
        for key, icon, label in nav_items:
            btn = self._nav_button(self.sidebar, icon, label, command=lambda k=key: self.show_page(k))
            self.nav_btns[key] = btn

        # --- Session Status Section ---
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=15, pady=15)
        tk.Label(self.sidebar, text="SESSION", font=("Courier New", 7, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(0, 8))
        self.status_dot = tk.Label(self.sidebar, text="● IDLE", font=("Courier New", 9, "bold"), fg=TEXT_DIM, bg=BG_PANEL)
        self.status_dot.pack(anchor="w", padx=18)
        self.sidebar_status = tk.Label(self.sidebar, text="No active operation", font=("Courier New", 8), fg=TEXT_SECONDARY, bg=BG_PANEL, wraplength=180, justify="left")
        self.sidebar_status.pack(anchor="w", padx=18, pady=(4, 0))

        # --- MATCH COUNTER SECTION (FIXES THE ERROR) ---
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=15, pady=15)
        tk.Label(self.sidebar, text="MATCHES FOUND", font=("Courier New", 7, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(0, 4))
        
        self.match_count_lbl = tk.Label(self.sidebar, text="0", font=("Courier New", 28, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL)
        self.match_count_lbl.pack(anchor="w", padx=18)

        # --- View Results Button ---
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=15, pady=15)
        self.view_results_btn = tk.Button(self.sidebar, text="▶  VIEW RESULTS", font=self.font_btn, bg=ACCENT_CYAN, fg=BG_DARK, relief="flat", pady=8, cursor="hand2", command=lambda: self.show_page("Results"))
        # We don't pack it yet; it will be shown by _on_search_complete when matches exist

    def _nav_button(self, parent, icon, label, command):
        frame = tk.Frame(parent, bg=BG_PANEL, cursor="hand2")
        frame.pack(fill="x", padx=10, pady=2)
        icon_l = tk.Label(frame, text=icon, font=("Courier New", 12), fg=TEXT_SECONDARY, bg=BG_PANEL, width=3)
        icon_l.pack(side="left")
        text_l = tk.Label(frame, text=label, font=("Courier New", 11), fg=TEXT_SECONDARY, bg=BG_PANEL, anchor="w")
        text_l.pack(side="left", fill="x", expand=True, pady=8)
        
        for w in (frame, icon_l, text_l):
            w.bind("<Button-1>", lambda e, c=command: c())
        frame._icon, frame._text = icon_l, text_l
        return frame

    def _set_active_nav(self, key):
        for k, frame in self.nav_btns.items():
            active = (k == key)
            bg = BG_SURFACE if active else BG_PANEL
            frame.config(bg=bg)
            frame._icon.config(bg=bg, fg=ACCENT_CYAN if active else TEXT_SECONDARY)
            frame._text.config(bg=bg, fg=TEXT_PRIMARY if active else TEXT_SECONDARY)

    # ------------------------------------------------------------------ #
    #  ANALYTICS PAGE UI                                                 #
    # ------------------------------------------------------------------ #
    def _build_analytics_page(self):
        p = self.analytics_page
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="PEOPLE COUNTING & HEATMAPS", font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        main_cols = tk.Frame(p, bg=BG_DARK)
        main_cols.pack(fill="both", expand=True, padx=25, pady=15)

        # --- RIGHT SIDEBAR (Control Panel) ---
        # Pack FIRST
        right_col = tk.Frame(main_cols, bg=BG_DARK, width=320)
        right_col.pack(side="right", fill="y", padx=(20, 0))
        right_col.pack_propagate(False)

        # (Existing widgets inside right_col)
        tk.Label(right_col, text="DATA SOURCE", font=self.font_head, fg=ACCENT_CYAN, bg=BG_DARK).pack(pady=(0, 10))
        tk.Button(right_col, text="📁 LOAD VIDEO FILE", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY, pady=10, command=self.select_analytics_video).pack(fill="x", pady=5)
        tk.Button(right_col, text="📷 START WEBCAM", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY, pady=10, command=lambda: self.start_analytics(0)).pack(fill="x", pady=5)

        tk.Frame(right_col, bg=BORDER, height=1).pack(fill="x", pady=20)
        self.ana_count_lbl = tk.Label(right_col, text="CURRENT: 0", font=("Courier New", 14, "bold"), fg=ACCENT_GREEN, bg=BG_PANEL, pady=15)
        self.ana_count_lbl.pack(fill="x", pady=5)
        self.ana_peak_lbl = tk.Label(right_col, text="PEAK: 0", font=("Courier New", 12), fg=TEXT_SECONDARY, bg=BG_PANEL, pady=15)
        self.ana_peak_lbl.pack(fill="x", pady=5)

        self.stop_ana_btn = tk.Button(right_col, text="■ STOP & SAVE REPORT", font=self.font_btn, bg=ACCENT_RED, fg=TEXT_PRIMARY, pady=15, state="disabled", command=self.stop_analytics)
        self.stop_ana_btn.pack(side="bottom", fill="x", pady=20)

        # --- LEFT AREA (Video Feed) ---
        # Pack SECOND
        left_col = tk.Frame(main_cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        ana_feed_wrap = tk.Frame(left_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        ana_feed_wrap.pack(fill="both", expand=True)
        self.ana_vid_label = tk.Label(ana_feed_wrap, bg="#040810", text="[ SELECT INPUT SOURCE ]", font=self.font_head, fg=TEXT_DIM)
        self.ana_vid_label.pack(fill="both", expand=True, padx=4, pady=4)
    
    def _apply_image_to_label(self, cv_frame, label):
        """Standardized image update for all video labels."""
        # Detect current size of the label widget
        lw = label.winfo_width()
        lh = label.winfo_height()
        
        # Fallback for initial state before window is rendered
        if lw < 10 or lh < 10:
            lw, lh = 800, 450
            
        img = Image.fromarray(cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)).resize((lw, lh))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk, text="")
        label._img_tk = img_tk

    # ------------------------------------------------------------------ #
    #  ANALYTICS LOGIC                                                   #
    # ------------------------------------------------------------------ #
    def select_analytics_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.start_analytics(file_path)

    def start_analytics(self, source):
        self.is_running = True
        self.stop_ana_btn.config(state="normal")
        self._set_sidebar_status("searching", "Analytics session active")
        threading.Thread(target=self.run_analytics_loop, args=(source,), daemon=True).start()

    def run_analytics_loop(self, source):
        cap = cv2.VideoCapture(source)
        ret, first_frame = cap.read()
        if not ret:
            self.is_running = False
            return
        
        self.analytics_engine.reset(first_frame.shape)
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break

            # Keep a copy of the last frame (backwards compatible name + new one)
            self.last_valid_frame = frame.copy()
            self.last_ana_frame = frame.copy()

            processed, count = self.analytics_engine.process_analytics_frame(frame)

            # Update stats on the main thread
            self.after(0, lambda c=count: self.ana_count_lbl.config(text=f"CURRENT: {c}"))
            self.after(0, lambda p=self.analytics_engine.max_count: self.ana_peak_lbl.config(text=f"PEAK: {p}"))

            # Update video display using centralized helper
            self.after(0, self._apply_image_to_label, processed, self.ana_vid_label)
        
        cap.release()

    def _update_ana_display(self, frame, count):
        self.ana_curr_lbl.config(text=f"CURRENT: {count}")
        self.ana_peak_lbl.config(text=f"PEAK: {self.analytics_engine.max_count}")
        
        lw, lh = self.ana_vid_label.winfo_width(), self.ana_vid_label.winfo_height()
        if lw < 10: lw, lh = 800, 450
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((lw, lh))
        img_tk = ImageTk.PhotoImage(img)
        self.ana_vid_label.config(image=img_tk, text="")
        self.ana_vid_label._img_tk = img_tk

    def stop_analytics(self):
        self.is_running = False
        self.stop_ana_btn.config(state="disabled")
        self._set_sidebar_status("done", "Analytics report generated")

        # Create final heatmap
        if hasattr(self, 'last_valid_frame'):
            final_heatmap = self.analytics_engine.get_final_heatmap(self.last_valid_frame)
            save_path = os.path.join(ANALYTICS_DIR, f"heatmap_{int(time.time())}.png")
            cv2.imwrite(save_path, final_heatmap)
            
            # Show final result in a new window or as a popup
            self.show_final_heatmap_popup(final_heatmap, save_path)

    def show_final_heatmap_popup(self, cv_img, path):
        top = tk.Toplevel(self)
        top.title("Session Report - People Density")
        top.configure(bg=BG_DARK)
        
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        img.thumbnail((1000, 700))
        img_tk = ImageTk.PhotoImage(img)
        
        lbl = tk.Label(top, image=img_tk, bg=BG_DARK)
        lbl.image = img_tk
        lbl.pack(padx=20, pady=20)
        
        tk.Label(top, text=f"Report saved to: {path}", fg=ACCENT_GREEN, bg=BG_DARK, font=self.font_mono).pack(pady=(0, 20))

    # ------------------------------------------------------------------ #
    #  EXISTING MODULES (RE-ID, HOME, RESULTS)                           #
    # ------------------------------------------------------------------ #
    
    
    def _build_home_page(self):
        p = self.home_page

        # --- Header Section ---
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=30, pady=(30, 0))
        tk.Label(hdr, text="SYSTEM COMMAND CENTER",
                 font=("Courier New", 22, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(anchor="w")
        tk.Label(hdr, text="Multi-module AI Surveillance & Behavioral Intelligence Platform",
                 font=("Courier New", 10), fg=ACCENT_CYAN, bg=BG_DARK).pack(anchor="w", pady=(4, 0))
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=30, pady=20)

        # --- Module Overview Cards ---
        cards_row = tk.Frame(p, bg=BG_DARK)
        cards_row.pack(fill="x", padx=30, pady=(0, 20))

        modules = [
            ("PERSON RE-ID", "Identify and track specific targets across multiple camera feeds.", ACCENT_BLUE, "◉"),
            ("HEATMAP ANALYTICS", "Visualize crowd density, traffic flow, and loitering hotspots.", ACCENT_AMBER, "📊"),
            ("RESTRICTED ZONES", "Define virtual boundaries and trigger instant intrusion alerts.", ACCENT_RED, "🚫"),
            ("SUSPECT FINDER",   "Search footage for a person matching a natural language description.", ACCENT_BLUE, "🔍"),
        ]

        for title, desc, color, icon in modules:
            card = tk.Frame(cards_row, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
            card.pack(side="left", padx=8, pady=4, fill="both", expand=True)
            
            inner = tk.Frame(card, bg=BG_CARD)
            inner.pack(padx=20, pady=20)
            
            tk.Label(inner, text=icon, font=("Courier New", 28), fg=color, bg=BG_CARD).pack(anchor="w")
            tk.Label(inner, text=title, font=("Courier New", 13, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD).pack(anchor="w", pady=(10, 5))
            tk.Label(inner, text=desc, font=("Courier New", 9), fg=TEXT_SECONDARY, bg=BG_CARD, wraplength=250, justify="left").pack(anchor="w")

        # --- Quick Launch Section ---
        launch_section = tk.Frame(p, bg=BG_DARK)
        launch_section.pack(fill="both", expand=True, padx=30, pady=10)
        
        tk.Label(launch_section, text="QUICK LAUNCH MODULES", font=("Courier New", 11, "bold"),
                 fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w", pady=(0, 15))

        launch_grid = tk.Frame(launch_section, bg=BG_DARK)
        launch_grid.pack(fill="x")

        # Launch Buttons Data
        buttons = [
            ("Launch Re-ID Engine", "ReID", ACCENT_BLUE),
            ("Open Heatmap Dashboard", "Analytics", ACCENT_AMBER),
            ("Configure Security Zones", "Zones", ACCENT_RED),
            ("Suspect Finder", "Suspect", ACCENT_BLUE),
            ("Anomaly Detection", "Anomaly", ACCENT_GREEN),
            ("View Search Results", "Results", TEXT_SECONDARY)
        ]

        # Arrange buttons in a 2x2 grid
        for i, (text, page, color) in enumerate(buttons):
            row, col = divmod(i, 2)
            btn_frame = tk.Frame(launch_grid, bg=BG_DARK)
            btn_frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            launch_grid.columnconfigure(col, weight=1)

            tk.Button(btn_frame, text=f"  {text}  →",
                      font=("Courier New", 11, "bold"),
                      bg=BG_SURFACE, fg=color,
                      activebackground=color, activeforeground=BG_DARK,
                      relief="flat", cursor="hand2", pady=15, anchor="w",
                      highlightbackground=BORDER, highlightthickness=1,
                      command=lambda p=page: self.show_page(p)).pack(fill="x")

        # --- System Specs Footer ---
        footer = tk.Frame(p, bg=BG_DARK)
        footer.pack(side="bottom", fill="x", padx=30, pady=20)
        specs = f"ENGINE: YOLOv8n | BACKBONE: OSNet x0_75 | DEVICE: {self.engine.device.upper()}"
        tk.Label(footer, text=specs, font=("Courier New", 8), fg=TEXT_DIM, bg=BG_DARK).pack(side="left")
        tk.Label(footer, text="v2.1.0-STABLE", font=("Courier New", 8), fg=TEXT_DIM, bg=BG_DARK).pack(side="right")
        
    # ------------------------------------------------------------------ #
    #  RE-ID PAGE                                                          #
    # ------------------------------------------------------------------ #
    def _build_reid_page(self):
        p = self.reid_page
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="PERSON RE-IDENTIFICATION", font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        self.phase_lbl = tk.Label(hdr, text="[ PHASE 1: TARGET SELECTION ]", font=("Courier New", 10, "bold"), fg=ACCENT_AMBER, bg=BG_DARK)
        self.phase_lbl.pack(side="right")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        cols = tk.Frame(p, bg=BG_DARK)
        cols.pack(fill="both", expand=True, padx=25, pady=15)

        # --- RIGHT SIDEBAR (Control Panel) ---
        right_col = tk.Frame(cols, bg=BG_DARK, width=320) 
        right_col.pack(side="right", fill="y", padx=(15, 0))
        right_col.pack_propagate(False) 

        # Operation Status
        status_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        status_card.pack(fill="x", pady=(0, 12))
        tk.Label(status_card, text="OPERATION STATUS", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.status_lbl = tk.Label(status_card, text="Awaiting start...", font=self.font_sub, fg=TEXT_SECONDARY, bg=BG_CARD, wraplength=280)
        self.status_lbl.pack(pady=10, padx=10)

        # Progress Bar
        prog_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        prog_card.pack(fill="x", pady=(0, 12))
        tk.Label(prog_card, text="SEARCH PROGRESS", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.prog_bar = ttk.Progressbar(prog_card, orient="horizontal", mode="determinate")
        self.prog_bar.pack(fill="x", padx=15, pady=10)
        self.prog_pct_lbl = tk.Label(prog_card, text="0%", font=self.font_mono, fg=TEXT_DIM, bg=BG_CARD)
        self.prog_pct_lbl.pack(pady=(0, 5))

        # Matches Count (THIS WAS MISSING)
        match_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        match_card.pack(fill="x", pady=(0, 12))
        tk.Label(match_card, text="MATCHES DETECTED", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.live_match_lbl = tk.Label(match_card, text="0", font=("Courier New", 26, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.live_match_lbl.pack(pady=10)

        # Tracked ID
        lock_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        lock_card.pack(fill="x", pady=(0, 12))
        tk.Label(lock_card, text="TRACKED ID", font=self.font_mono, fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.locked_id_lbl = tk.Label(lock_card, text="—", font=self.font_huge, fg=ACCENT_AMBER, bg=BG_CARD)
        self.locked_id_lbl.pack(pady=10)

        # Button Frame for Start/Stop
        btn_frame = tk.Frame(right_col, bg=BG_DARK)
        btn_frame.pack(fill="x", side="bottom", pady=(10, 0))

        self.stop_reid_btn = tk.Button(btn_frame, text="■ STOP REID", font=self.font_btn, bg=ACCENT_RED, fg=TEXT_PRIMARY, pady=12, state="disabled", command=self.stop_reid)
        self.stop_reid_btn.pack(fill="x", pady=(0, 8))

        self.start_btn = tk.Button(btn_frame, text="▶ START PHASE 1", font=self.font_btn, bg=ACCENT_GREEN, fg=BG_DARK, pady=12, command=self.start_p1)
        self.start_btn.pack(fill="x")

        # --- LEFT AREA (Video Feed) ---
        left_col = tk.Frame(cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        feed_wrap = tk.Frame(left_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        feed_wrap.pack(fill="both", expand=True)
        self.vid_label = tk.Label(feed_wrap, bg="#040810", text="[ VIDEO FEED INACTIVE ]", font=self.font_title, fg=TEXT_DIM)
        self.vid_label.pack(fill="both", expand=True, padx=4, pady=4)
        self.vid_label.bind("<Button-1>", self.handle_click)

    def _update_gallery_bars(self):
        count = len(self.engine.target_gallery)
        for i, bar in enumerate(self.gallery_bars):
            bar.config(bg=ACCENT_CYAN if i < count else TEXT_DIM)
        self.gallery_count_lbl.config(text=f"{count} / 15 samples")
        if count > 0:
            self.gallery_count_lbl.config(fg=ACCENT_CYAN)
        self.after(200, self._update_gallery_bars)

    # ------------------------------------------------------------------ #
    #  RESULTS PAGE  (multi-tab: ReID · Zones · Heatmaps)                 #
    # ------------------------------------------------------------------ #
    def _build_results_page(self):
        p = self.results_page

        # ── Page header ──────────────────────────────────────────────────
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 0))
        tk.Label(hdr, text="RESULTS ARCHIVE",
                 font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        self.result_counter_lbl = tk.Label(hdr, text="", font=("Courier New", 10),
                                            fg=TEXT_SECONDARY, bg=BG_DARK)
        self.result_counter_lbl.pack(side="right")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25, pady=(10, 0))

        # ── Tab bar ──────────────────────────────────────────────────────
        self._active_results_tab = "reid"   # "reid" | "zones" | "heatmaps"

        tab_bar = tk.Frame(p, bg=BG_PANEL, height=42)
        tab_bar.pack(fill="x", padx=25, pady=(0, 0))
        tab_bar.pack_propagate(False)

        self._tab_btns = {}
        tab_defs = [
            ("reid",     "◉  PERSON RE-ID",     ACCENT_CYAN),
            ("zones",    "🚫  ZONE ALERTS",      ACCENT_RED),
            ("heatmaps", "📊  HEATMAP REPORTS",  ACCENT_AMBER),
            ("suspect",  "🔍  SUSPECT FINDER",   ACCENT_BLUE),
            ("anomaly",  "⚠  ANOMALY ALERTS",   ACCENT_AMBER),
        ]
        for key, label, color in tab_defs:
            btn = tk.Button(tab_bar, text=label,
                            font=("Courier New", 10, "bold"),
                            bg=BG_SURFACE if key == "reid" else BG_PANEL,
                            fg=color if key == "reid" else TEXT_SECONDARY,
                            relief="flat", cursor="hand2",
                            padx=20, pady=10,
                            command=lambda k=key: self._switch_results_tab(k))
            btn.pack(side="left")
            self._tab_btns[key] = btn

        # Refresh button on the right
        tk.Button(tab_bar, text="↺  REFRESH FROM DISK",
                  font=("Courier New", 9, "bold"),
                  bg=BG_PANEL, fg=TEXT_DIM,
                  relief="flat", cursor="hand2",
                  padx=14, pady=10,
                  command=self._refresh_results_page).pack(side="right")

        tk.Frame(p, bg=BORDER_BRIGHT, height=1).pack(fill="x", padx=25)

        # ── Main content ─────────────────────────────────────────────────
        main = tk.Frame(p, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=25, pady=12)

        # Left: image / report viewer ────────────────────────────────────
        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        img_card = tk.Frame(left, bg=BG_CARD, highlightbackground=BORDER_BRIGHT,
                            highlightthickness=1)
        img_card.pack(fill="both", expand=True)

        img_topbar = tk.Frame(img_card, bg=BG_SURFACE)
        img_topbar.pack(fill="x")
        self.result_preview_title = tk.Label(img_topbar, text="▦  MATCH PREVIEW",
                 font=("Courier New", 9, "bold"), fg=ACCENT_CYAN, bg=BG_SURFACE)
        self.result_preview_title.pack(side="left", padx=12, pady=6)

        self.result_img_lbl = tk.Label(img_card, bg="#040810",
                                        text="[ SELECT A TAB TO VIEW RESULTS ]",
                                        font=("Courier New", 14), fg=TEXT_DIM)
        self.result_img_lbl.pack(fill="both", expand=True, padx=8, pady=8)

        # Zoom bindings
        self.result_img_lbl.bind("<Key-plus>",    lambda e: self.zoom_in_result())
        self.result_img_lbl.bind("<Key-minus>",   lambda e: self.zoom_out_result())
        self.result_img_lbl.bind("<Key-equal>",   lambda e: self.zoom_in_result())
        self.result_img_lbl.bind("<MouseWheel>",  self._on_results_mousewheel)
        self.result_img_lbl.bind("<Button-4>",    self._on_results_mousewheel)
        self.result_img_lbl.bind("<Button-5>",    self._on_results_mousewheel)

        # Nav bar
        nav_bar = tk.Frame(left, bg=BG_DARK)
        nav_bar.pack(fill="x", pady=(10, 0))

        self.prev_btn = tk.Button(nav_bar, text="◀  PREVIOUS",
                                   font=("Courier New", 11, "bold"),
                                   bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat",
                                   cursor="hand2", pady=10, padx=20,
                                   highlightbackground=BORDER, highlightthickness=1,
                                   command=self.show_prev_result)
        self.prev_btn.pack(side="left")

        self.nav_pos_lbl = tk.Label(nav_bar, text="0 / 0",
                                     font=("Courier New", 12, "bold"),
                                     fg=TEXT_SECONDARY, bg=BG_DARK)
        self.nav_pos_lbl.pack(side="left", expand=True)

        self.next_btn = tk.Button(nav_bar, text="NEXT  ▶",
                                   font=("Courier New", 11, "bold"),
                                   bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat",
                                   cursor="hand2", pady=10, padx=20,
                                   highlightbackground=BORDER, highlightthickness=1,
                                   command=self.show_next_result)
        self.next_btn.pack(side="right")

        # Zoom bar
        zoom_bar = tk.Frame(left, bg=BG_DARK)
        zoom_bar.pack(fill="x", pady=(8, 0))

        tk.Label(zoom_bar, text="ZOOM:", font=("Courier New", 10, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).pack(side="left", padx=5)

        self.zoom_out_btn = tk.Button(zoom_bar, text="−  ZOOM OUT",
                                        font=("Courier New", 10, "bold"),
                                        bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat",
                                        cursor="hand2", pady=8, padx=15,
                                        highlightbackground=BORDER, highlightthickness=1,
                                        command=self.zoom_out_result)
        self.zoom_out_btn.pack(side="left", padx=5)

        self.zoom_lbl = tk.Label(zoom_bar, text="100%",
                                  font=("Courier New", 10, "bold"),
                                  fg=ACCENT_CYAN, bg=BG_DARK, width=8)
        self.zoom_lbl.pack(side="left", padx=5)

        self.zoom_in_btn = tk.Button(zoom_bar, text="ZOOM IN  +",
                                       font=("Courier New", 10, "bold"),
                                       bg=BG_CARD, fg=TEXT_PRIMARY, relief="flat",
                                       cursor="hand2", pady=8, padx=15,
                                       highlightbackground=BORDER, highlightthickness=1,
                                       command=self.zoom_in_result)
        self.zoom_in_btn.pack(side="left", padx=5)

        self.reset_zoom_btn = tk.Button(zoom_bar, text="RESET",
                                          font=("Courier New", 10, "bold"),
                                          bg=BG_CARD, fg=TEXT_SECONDARY, relief="flat",
                                          cursor="hand2", pady=8, padx=12,
                                          highlightbackground=BORDER, highlightthickness=1,
                                          command=self.reset_zoom)
        self.reset_zoom_btn.pack(side="left", padx=5)

        # Right: metadata panel ──────────────────────────────────────────
        right = tk.Frame(main, bg=BG_DARK, width=300)
        right.pack(side="right", fill="y", padx=(20, 0))
        right.pack_propagate(False)

        # ── Tab summary card ─────────────────────────────────────────────
        tab_summary = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER,
                               highlightthickness=1)
        tab_summary.pack(fill="x", pady=(0, 12))
        tk.Label(tab_summary, text="MODULE TOTALS",
                 font=("Courier New", 9, "bold"), fg=TEXT_SECONDARY,
                 bg=BG_SURFACE).pack(fill="x", padx=12, pady=6)

        ts_inner = tk.Frame(tab_summary, bg=BG_CARD)
        ts_inner.pack(fill="x", padx=12, pady=8)

        self._tab_count_labels = {}
        tab_count_defs = [
            ("reid",     "◉ PERSON RE-ID",    ACCENT_CYAN),
            ("zones",    "🚫 ZONE ALERTS",     ACCENT_RED),
            ("heatmaps", "📊 HEATMAP REPORTS", ACCENT_AMBER),
            ("suspect",  "🔍 SUSPECT FINDER",  ACCENT_BLUE),
            ("anomaly",  "⚠ ANOMALY ALERTS",  ACCENT_AMBER),
        ]
        for key, label, color in tab_count_defs:
            row = tk.Frame(ts_inner, bg=BG_CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, font=("Courier New", 8, "bold"),
                     fg=color, bg=BG_CARD).pack(side="left")
            cnt = tk.Label(row, text="0", font=("Courier New", 10, "bold"),
                           fg=TEXT_PRIMARY, bg=BG_CARD)
            cnt.pack(side="right")
            self._tab_count_labels[key] = cnt

        # ── Match metadata card ──────────────────────────────────────────
        meta_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER,
                             highlightthickness=1)
        meta_card.pack(fill="x", pady=(0, 12))
        tk.Label(meta_card, text="ITEM METADATA",
                 font=("Courier New", 9, "bold"), fg=TEXT_SECONDARY,
                 bg=BG_SURFACE).pack(fill="x", padx=12, pady=6)

        meta_inner = tk.Frame(meta_card, bg=BG_CARD)
        meta_inner.pack(fill="x", padx=12, pady=12)

        self.meta_fields = {}
        for key, label in [("source",    "SOURCE"),
                            ("timestamp", "TIMESTAMP"),
                            ("index",     "ITEM INDEX"),
                            ("saved",     "FILE NAME"),
                            ("type",      "RESULT TYPE")]:
            tk.Label(meta_inner, text=label, font=("Courier New", 8, "bold"),
                     fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", pady=(6, 0))
            val_lbl = tk.Label(meta_inner, text="—", font=("Courier New", 10),
                               fg=TEXT_PRIMARY, bg=BG_CARD, wraplength=240,
                               justify="left", anchor="w")
            val_lbl.pack(anchor="w")
            self.meta_fields[key] = val_lbl

        # Backwards-compatible alias used elsewhere in code
        self.meta_fields["video"] = self.meta_fields["source"]

        # ── Session summary (ReID-specific) ──────────────────────────────
        summary_card = tk.Frame(right, bg=BG_CARD, highlightbackground=BORDER,
                                highlightthickness=1)
        summary_card.pack(fill="x", pady=(0, 12))
        tk.Label(summary_card, text="SESSION SUMMARY",
                 font=("Courier New", 9, "bold"), fg=TEXT_SECONDARY,
                 bg=BG_SURFACE).pack(fill="x", padx=12, pady=6)

        sum_inner = tk.Frame(summary_card, bg=BG_CARD)
        sum_inner.pack(fill="x", padx=12, pady=12)

        for key, label in [("total_matches", "TOTAL MATCHES"),
                            ("save_dir",      "SAVE DIRECTORY"),
                            ("threshold",     "MATCH THRESHOLD")]:
            tk.Label(sum_inner, text=label, font=("Courier New", 8, "bold"),
                     fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", pady=(6, 0))
            val = {"total_matches": "0",
                   "save_dir":      f"./{MATCHES_DIR}/",
                   "threshold":     "> 0.75 cosine sim"}.get(key, "—")
            lbl = tk.Label(sum_inner, text=val, font=("Courier New", 10),
                           fg=TEXT_PRIMARY, bg=BG_CARD, anchor="w")
            lbl.pack(anchor="w")
            if key == "total_matches":
                self.results_total_lbl = lbl

        # Back button
        tk.Button(right, text="◀  BACK TO TRACKING",
                  font=("Courier New", 10, "bold"),
                  bg=BG_CARD, fg=TEXT_SECONDARY, relief="flat",
                  cursor="hand2", pady=10,
                  highlightbackground=BORDER, highlightthickness=1,
                  command=lambda: self.show_page("ReID")).pack(fill="x")

    # ── Tab switching ────────────────────────────────────────────────────
    def _switch_results_tab(self, tab_key):
        """Switch active results tab and reload items."""
        self._active_results_tab = tab_key
        tab_colors = {"reid": ACCENT_CYAN, "zones": ACCENT_RED, "heatmaps": ACCENT_AMBER, "suspect": ACCENT_BLUE, "anomaly": ACCENT_AMBER}

        for k, btn in self._tab_btns.items():
            active = (k == tab_key)
            btn.config(bg=BG_SURFACE if active else BG_PANEL,
                       fg=tab_colors[k] if active else TEXT_SECONDARY)

        # Update preview title accent
        self.result_preview_title.config(fg=tab_colors.get(tab_key, ACCENT_CYAN))

        self._load_tab_items(tab_key)

    def _load_tab_items(self, tab_key):
        """Scan the appropriate directory and populate self._active_items."""
        if tab_key == "reid":
            items = self._scan_reid_results()
            type_label = "Person Re-ID Match"
            dir_label = f"./{MATCHES_DIR}/"
        elif tab_key == "zones":
            items = self._scan_zone_results()
            type_label = "Zone Intrusion Alert"
            dir_label = f"./{ZONE_ALERTS_DIR}/"
        elif tab_key == "suspect":
            items = self._scan_suspect_results()
            type_label = "Suspect Finder Match"
            dir_label = f"./{SUSPECT_FINDER_DIR}/"
        elif tab_key == "anomaly":
            items = self._scan_anomaly_results()
            type_label = "Anomaly Detection Alert"
            dir_label = "./evidence_clips/"
        else:  # heatmaps
            items = self._scan_heatmap_results()
            type_label = "Heatmap / Analytics Report"
            dir_label = f"./{ANALYTICS_DIR}/"

        self._active_items = items  # list of (filepath, source_label, timestamp_str)
        self.current_result_idx = 0

        total = len(items)
        self._tab_count_labels[tab_key].config(text=str(total))
        self.result_counter_lbl.config(text=f"{total} result(s)")
        self.results_total_lbl.config(text=str(total))
        self.meta_fields["type"].config(text=type_label)

        if total == 0:
            self.result_img_lbl.config(image="", text=f"[ NO {tab_key.upper()} RESULTS FOUND ]")
            self.nav_pos_lbl.config(text="0 / 0")
            for k, f in self.meta_fields.items():
                if k not in ("type",):
                    f.config(text="—")
            self.zoom_lbl.config(text="—")
            return

        self._show_active_item(0)
        self.result_img_lbl.focus_set()

    # ── Directory scanners ───────────────────────────────────────────────
    def _scan_reid_results(self):
        """Return in-memory matches first; fall back to disk scan."""
        if self.match_results:
            return list(self.match_results)   # (filepath, video_name, timestamp)
        return self._scan_images_in_dir(MATCHES_DIR)

    def _scan_zone_results(self):
        """Recursively find all zone alert images, paired with their JSON."""
        items = []
        if not os.path.isdir(ZONE_ALERTS_DIR):
            return items
        for root, dirs, files in os.walk(ZONE_ALERTS_DIR):
            for fname in sorted(files):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    fpath = os.path.join(root, fname)
                    # Try to load accompanying JSON for metadata
                    json_path = fpath.rsplit(".", 1)[0] + ".json"
                    ts = "—"
                    source = os.path.basename(root)
                    if os.path.isfile(json_path):
                        try:
                            import json as _json
                            with open(json_path) as jf:
                                meta = _json.load(jf)
                            ts = meta.get("timestamp", ts)
                            source = meta.get("alert_message", source)[:40]
                        except Exception:
                            pass
                    items.append((fpath, source, ts))
        return items

    def _scan_heatmap_results(self):
        """Return all PNG/JPG files in analytics_output."""
        return self._scan_images_in_dir(ANALYTICS_DIR)

    def _scan_suspect_results(self):
        """Return in-memory suspect matches first; fall back to disk scan of suspect_results/."""
        if hasattr(self, 'suspect_match_results') and self.suspect_match_results:
            items = []
            for r in self.suspect_match_results:
                # r is a MatchResult from suspect_finder.py
                # Reconstruct the saved filepath
                fname = f"suspect_{self.suspect_match_results.index(r)+1:04d}_f{r.frame_num}_s{r.overall_score:.2f}.jpg"
                fpath = os.path.join(SUSPECT_FINDER_DIR, fname)
                from datetime import timedelta
                ts = str(timedelta(seconds=int(r.timestamp)))
                score_str = f"Score {r.overall_score:.0%}  |  Frame {r.frame_num}  |  {ts}"
                items.append((fpath, score_str, ts))
            # Filter to only files that actually exist on disk
            items = [(fp, src, ts) for fp, src, ts in items if os.path.isfile(fp)]
            if items:
                return items
        return self._scan_images_in_dir(SUSPECT_FINDER_DIR)

    def _scan_images_in_dir(self, directory):
        """Generic scan: returns list of (filepath, dir_name, mtime_str)."""
        items = []
        if not os.path.isdir(directory):
            return items
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(directory, fname)
                mtime = os.path.getmtime(fpath)
                ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                items.append((fpath, directory, ts))
        return items

    def _scan_anomaly_results(self):
        """Scan evidence_clips for anomaly images and video clips (first frame)."""
        items = []
        evidence_dir = "evidence_clips"
        if not os.path.isdir(evidence_dir):
            return items
        thumb_dir = os.path.join(evidence_dir, ".thumbs")
        os.makedirs(thumb_dir, exist_ok=True)
        for root, dirs, files in os.walk(evidence_dir):
            # Skip the thumbs directory itself
            if ".thumbs" in root:
                continue
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                mtime = os.path.getmtime(fpath)
                ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                event_type = os.path.basename(root).upper() if os.path.basename(root) != "evidence_clips" else "ANOMALY"
                label = f"{event_type} — {fname}"
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((fpath, label, ts))
                elif fname.lower().endswith((".avi", ".mp4", ".mkv")):
                    # Extract first frame as thumbnail
                    thumb_path = os.path.join(thumb_dir, fname + ".jpg")
                    if not os.path.isfile(thumb_path):
                        try:
                            cap_t = cv2.VideoCapture(fpath)
                            ret_t, frame_t = cap_t.read()
                            if ret_t:
                                # Add a label overlay
                                cv2.putText(frame_t, f"▶ {event_type}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                                cv2.imwrite(thumb_path, frame_t)
                            cap_t.release()
                        except Exception:
                            pass
                    if os.path.isfile(thumb_path):
                        items.append((thumb_path, label, ts))
        return items

    # ------------------------------------------------------------------ #
    #  ANOMALY DETECTION PAGE                                             #
    # ------------------------------------------------------------------ #
    # ================================================================== #
    #  ANOMALY DETECTION PAGE                                           #
    # ================================================================== #
    def _build_anomaly_page(self):
        p = self.anomaly_page

        # ── Page header ────────────────────────────────────────────────
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="ANOMALY DETECTION",
                 font=("Courier New", 16, "bold"), fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        self.anomaly_phase_lbl = tk.Label(hdr, text="[ IDLE ]",
                                           font=("Courier New", 10, "bold"),
                                           fg=ACCENT_AMBER, bg=BG_DARK)
        self.anomaly_phase_lbl.pack(side="right")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        # ── Instruction banner (shown during zone-draw mode) ───────────
        self.anomaly_instruction_bar = tk.Frame(p, bg="#1a1000", height=34)
        self.anomaly_instruction_lbl = tk.Label(
            self.anomaly_instruction_bar,
            text="✦  ZONE DRAW MODE  —  Click to add points  |  ENTER = confirm zone  |  C = clear  |  ESC = full-frame",
            font=("Courier New", 9, "bold"), fg=ACCENT_AMBER, bg="#1a1000")
        self.anomaly_instruction_lbl.pack(side="left", padx=15, pady=8)
        # Don't pack yet — shown only in draw mode

        # ── Main column layout ─────────────────────────────────────────
        cols = tk.Frame(p, bg=BG_DARK)
        cols.pack(fill="both", expand=True, padx=25, pady=(10, 15))

        # ── RIGHT SIDEBAR ──────────────────────────────────────────────
        right_col = tk.Frame(cols, bg=BG_DARK, width=285)
        right_col.pack(side="right", fill="y", padx=(15, 0))
        right_col.pack_propagate(False)

        # Data Source
        src_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        src_card.pack(fill="x", pady=(0, 8))
        tk.Label(src_card, text="DATA SOURCE", font=("Courier New", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", padx=10, pady=6)
        src_inner = tk.Frame(src_card, bg=BG_CARD)
        src_inner.pack(fill="x", padx=8, pady=8)

        self.anomaly_source_lbl = tk.Label(src_inner, text="No source loaded",
                                            font=("Courier New", 8), fg=TEXT_DIM,
                                            bg=BG_CARD, wraplength=230, justify="left")
        self.anomaly_source_lbl.pack(anchor="w", pady=(0, 8))

        tk.Button(src_inner, text="📁  LOAD VIDEO FILE", font=self.font_btn,
                  bg=BG_SURFACE, fg=TEXT_PRIMARY, relief="flat", pady=8, cursor="hand2",
                  highlightbackground=BORDER, highlightthickness=1,
                  command=self._anomaly_load_video).pack(fill="x", pady=(0, 5))
        tk.Button(src_inner, text="📷  USE WEBCAM", font=self.font_btn,
                  bg=BG_SURFACE, fg=TEXT_PRIMARY, relief="flat", pady=8, cursor="hand2",
                  highlightbackground=BORDER, highlightthickness=1,
                  command=self._anomaly_use_webcam).pack(fill="x")

        # Zone Setup
        zone_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        zone_card.pack(fill="x", pady=(0, 8))
        tk.Label(zone_card, text="MONITORING ZONE", font=("Courier New", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", padx=10, pady=6)
        zone_inner = tk.Frame(zone_card, bg=BG_CARD)
        zone_inner.pack(fill="x", padx=8, pady=8)

        self.anomaly_zone_status = tk.Label(zone_inner,
                                             text="● Full frame (default)",
                                             font=("Courier New", 8, "bold"),
                                             fg=ACCENT_GREEN, bg=BG_CARD)
        self.anomaly_zone_status.pack(anchor="w", pady=(0, 8))

        zone_btn_row = tk.Frame(zone_inner, bg=BG_CARD)
        zone_btn_row.pack(fill="x")
        self.anomaly_draw_zone_btn = tk.Button(
            zone_btn_row, text="✏  DRAW ZONE", font=self.font_btn,
            bg=ACCENT_CYAN, fg=BG_DARK, relief="flat", pady=7, cursor="hand2",
            command=self._anomaly_enter_draw_mode)
        self.anomaly_draw_zone_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        tk.Button(zone_btn_row, text="✕  CLEAR", font=self.font_btn,
                  bg=BG_SURFACE, fg=ACCENT_RED, relief="flat", pady=7, cursor="hand2",
                  highlightbackground=BORDER, highlightthickness=1,
                  command=self._anomaly_clear_zone).pack(side="left")

        # Detection Counts
        cnt_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        cnt_card.pack(fill="x", pady=(0, 8))
        tk.Label(cnt_card, text="DETECTION COUNTS", font=("Courier New", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", padx=10, pady=6)
        cnt_inner = tk.Frame(cnt_card, bg=BG_CARD)
        cnt_inner.pack(fill="x", padx=10, pady=8)

        self.anomaly_counters = {}
        event_styles = [
            ("FALL",      ACCENT_RED,   "🔴"),

            ("CROWD",     ACCENT_AMBER, "🟡"),
            ("LOITERING", ACCENT_CYAN,  "🔵"),
            ("RUNNING",   ACCENT_AMBER, "🟠"),
        ]
        for event, color, icon in event_styles:
            row = tk.Frame(cnt_inner, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{icon} {event}", font=("Courier New", 9, "bold"),
                     fg=color, bg=BG_CARD).pack(side="left")
            cnt_lbl = tk.Label(row, text="0", font=("Courier New", 12, "bold"),
                               fg=TEXT_PRIMARY, bg=BG_CARD)
            cnt_lbl.pack(side="right")
            self.anomaly_counters[event] = cnt_lbl

        # Progress
        prog_card = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        prog_card.pack(fill="x", pady=(0, 8))
        tk.Label(prog_card, text="PROGRESS", font=("Courier New", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", padx=10, pady=6)
        prog_inner = tk.Frame(prog_card, bg=BG_CARD)
        prog_inner.pack(fill="x", padx=10, pady=8)
        self.anomaly_prog = ttk.Progressbar(prog_inner, orient="horizontal", mode="determinate")
        self.anomaly_prog.pack(fill="x")
        self.anomaly_prog_lbl = tk.Label(prog_inner, text="0%",
                                          font=("Courier New", 9), fg=TEXT_DIM, bg=BG_CARD)
        self.anomaly_prog_lbl.pack(anchor="e", pady=(2, 0))

        # Start / Stop  — must be packed BEFORE the expand=True log frame
        btn_row = tk.Frame(right_col, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(0, 4))

        self.anomaly_start_btn = tk.Button(
            btn_row, text="▶  START DETECTION", font=self.font_btn,
            bg=ACCENT_GREEN, fg=BG_DARK, relief="flat", pady=11,
            cursor="hand2", command=self._anomaly_start)
        self.anomaly_start_btn.pack(side="left", fill="x", expand=True)

        self.anomaly_stop_btn = tk.Button(
            btn_row, text="■  STOP", font=self.font_btn,
            bg=ACCENT_RED, fg=TEXT_PRIMARY, relief="flat", pady=11,
            state="disabled", cursor="hand2", command=self._anomaly_stop)
        self.anomaly_stop_btn.pack(side="right", padx=(6, 0), fill="x", expand=True)

        # View Results
        tk.Button(right_col, text="▦  VIEW ANOMALY RESULTS",
                  font=self.font_btn, bg=ACCENT_AMBER, fg=BG_DARK, pady=8,
                  relief="flat", cursor="hand2",
                  command=lambda: (self.show_page("Results"),
                                   self._switch_results_tab("anomaly"))).pack(fill="x", pady=(0, 6))

        # Security Log  — expand=True must come LAST so buttons above stay visible
        tk.Label(right_col, text="SECURITY LOG", font=("Courier New", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).pack(anchor="w", pady=(4, 2))
        log_frame = tk.Frame(right_col, bg=BG_CARD, highlightthickness=1,
                             highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True, pady=(0, 0))
        log_scroll = tk.Scrollbar(log_frame, orient="vertical",
                                   bg=BG_CARD, troughcolor=BG_PANEL)
        log_scroll.pack(side="right", fill="y")
        self.anomaly_log_listbox = tk.Listbox(
            log_frame, bg="#08090f", fg=TEXT_PRIMARY,
            font=("Courier New", 7), selectbackground=BG_SURFACE,
            highlightthickness=0, relief="flat",
            yscrollcommand=log_scroll.set, activestyle="none")
        self.anomaly_log_listbox.pack(fill="both", expand=True, padx=2)
        log_scroll.config(command=self.anomaly_log_listbox.yview)

        # ── LEFT AREA — Video canvas with zone overlay ─────────────────
        left_col = tk.Frame(cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)

        feed_wrap = tk.Frame(left_col, bg=BG_CARD,
                             highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        feed_wrap.pack(fill="both", expand=True)

        # Use a Canvas so we can draw the zone polygon overlay on top
        self.anomaly_canvas = tk.Canvas(feed_wrap, bg="#040810",
                                         highlightthickness=0, cursor="crosshair")
        self.anomaly_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Placeholder text on canvas
        self._anomaly_canvas_placeholder()

        # Canvas mouse bindings (active only in draw mode)
        self.anomaly_canvas.bind("<Button-1>",   self._anomaly_canvas_click)
        self.anomaly_canvas.bind("<Motion>",      self._anomaly_canvas_motion)
        self.anomaly_canvas.bind("<Key-Return>",  self._anomaly_canvas_confirm)
        self.anomaly_canvas.bind("<Return>",      self._anomaly_canvas_confirm)
        self.anomaly_canvas.bind("<Escape>",      self._anomaly_canvas_escape)
        self.anomaly_canvas.bind("<c>",           self._anomaly_canvas_clear_pts)
        self.anomaly_canvas.bind("<C>",           self._anomaly_canvas_clear_pts)
        self.anomaly_canvas.focus_set()
        # Also bind Return at the page level so it works even without canvas focus
        p.bind("<Return>", self._anomaly_canvas_confirm)
        p.bind("<Escape>", self._anomaly_canvas_escape)

        # Status bar below canvas
        status_bar = tk.Frame(left_col, bg=BG_DARK)
        status_bar.pack(fill="x", pady=(6, 0))

        self.anomaly_frame_lbl = tk.Label(status_bar, text="",
                                           font=("Courier New", 8), fg=TEXT_DIM, bg=BG_DARK)
        self.anomaly_frame_lbl.pack(side="left")

        self.anomaly_pts_lbl = tk.Label(status_bar, text="",
                                         font=("Courier New", 8, "bold"),
                                         fg=ACCENT_CYAN, bg=BG_DARK)
        self.anomaly_pts_lbl.pack(side="right")

        # ── Internal zone-draw state ────────────────────────────────────
        self.anomaly_draw_mode   = False      # True while user is drawing
        self.anomaly_zone_points = []         # canvas pixel coords [(x,y)...]
        self.anomaly_roi_polygon = None       # numpy array in video coords (set on confirm)
        self._anomaly_img_tk     = None       # keep PhotoImage reference
        self._anomaly_first_frame = None      # raw BGR frame for zone setup preview
        self._anomaly_mouse_pos  = (0, 0)    # live cursor position in draw mode

    # ── Canvas helpers ──────────────────────────────────────────────────
    def _anomaly_canvas_placeholder(self):
        self.anomaly_canvas.delete("all")
        cw = self.anomaly_canvas.winfo_width()  or 800
        ch = self.anomaly_canvas.winfo_height() or 450
        self.anomaly_canvas.create_text(
            cw // 2, ch // 2,
            text="[ SELECT INPUT SOURCE ]\n\nLoad a video or start webcam,\nthen draw your monitoring zone.",
            font=("Courier New", 13), fill="#3d4f6b",
            justify="center", tags="placeholder")

    def _anomaly_canvas_render_frame(self, bgr_frame):
        """Render a BGR frame onto the canvas, return scale factors."""
        cw = self.anomaly_canvas.winfo_width()  or 800
        ch = self.anomaly_canvas.winfo_height() or 450
        img = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        img = img.resize((cw, ch), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.anomaly_canvas.delete("all")
        self.anomaly_canvas.create_image(0, 0, anchor="nw", image=img_tk, tags="frame")
        self._anomaly_img_tk = img_tk
        fh, fw = bgr_frame.shape[:2]
        return cw / fw, ch / fh   # scale_x, scale_y

    def _anomaly_redraw_zone(self):
        """Redraw the zone polygon/points overlay on the canvas."""
        self.anomaly_canvas.delete("zone")
        pts = self.anomaly_zone_points
        if not pts:
            return
        mx, my = self._anomaly_mouse_pos
        # Draw filled translucent polygon if >= 3 pts
        if len(pts) >= 3:
            flat = [c for p in pts for c in p]
            self.anomaly_canvas.create_polygon(
                flat, fill="#00e5ff", stipple="gray25",
                outline="#00e5ff", width=2, tags="zone")
        # Draw edge lines (including preview line to cursor)
        for i in range(len(pts) - 1):
            self.anomaly_canvas.create_line(
                pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                fill="#00e5ff", width=2, tags="zone")
        if self.anomaly_draw_mode and len(pts) >= 1:
            self.anomaly_canvas.create_line(
                pts[-1][0], pts[-1][1], mx, my,
                fill="#00e5ff", width=1, dash=(4, 4), tags="zone")
            if len(pts) >= 2:
                self.anomaly_canvas.create_line(
                    pts[0][0], pts[0][1], mx, my,
                    fill="#00e5ff", width=1, dash=(4, 4), tags="zone")
        # Draw vertex dots
        for i, (px, py) in enumerate(pts):
            r = 6 if i == 0 else 4
            col = ACCENT_GREEN if i == 0 else ACCENT_CYAN
            self.anomaly_canvas.create_oval(
                px-r, py-r, px+r, py+r,
                fill=col, outline="white", width=1, tags="zone")
            self.anomaly_canvas.create_text(
                px+10, py-10,
                text=str(i+1), fill="white",
                font=("Courier New", 7, "bold"), tags="zone")

        # Point count label
        self.anomaly_pts_lbl.config(
            text=f"Points: {len(pts)}  |  ENTER to confirm  |  C to clear")

    # ── Canvas event handlers ───────────────────────────────────────────
    def _anomaly_canvas_click(self, event):
        if not self.anomaly_draw_mode:
            return
        self.anomaly_canvas.focus_set()
        self.anomaly_zone_points.append((event.x, event.y))
        self._anomaly_redraw_zone()

    def _anomaly_canvas_motion(self, event):
        if not self.anomaly_draw_mode:
            return
        self._anomaly_mouse_pos = (event.x, event.y)
        self._anomaly_redraw_zone()

    def _anomaly_canvas_confirm(self, event=None):
        """Confirm drawn zone and exit draw mode."""
        if not self.anomaly_draw_mode:
            return
        pts = self.anomaly_zone_points
        if len(pts) < 3:
            self.anomaly_zone_status.config(
                text="⚠ Need at least 3 points", fg=ACCENT_RED)
            return
        self._anomaly_commit_zone(pts)

    def _anomaly_canvas_escape(self, event=None):
        """ESC: use full-frame zone."""
        if not self.anomaly_draw_mode:
            return
        self.anomaly_zone_points = []
        self.anomaly_roi_polygon = None
        self._anomaly_exit_draw_mode()
        self.anomaly_zone_status.config(
            text="● Full frame (default)", fg=ACCENT_GREEN)
        self.anomaly_pts_lbl.config(text="")

    def _anomaly_canvas_clear_pts(self, event=None):
        if not self.anomaly_draw_mode:
            return
        self.anomaly_zone_points = []
        self._anomaly_redraw_zone()

    def _anomaly_commit_zone(self, canvas_pts):
        """Convert canvas pixel coords → video frame coords and store as ROI."""
        import numpy as np
        frame = self._anomaly_first_frame
        if frame is None:
            self.anomaly_roi_polygon = None
            self._anomaly_exit_draw_mode()
            return
        cw = self.anomaly_canvas.winfo_width()  or 800
        ch = self.anomaly_canvas.winfo_height() or 450
        fh, fw = frame.shape[:2]
        sx, sy = fw / cw, fh / ch
        vid_pts = [(int(x * sx), int(y * sy)) for x, y in canvas_pts]
        self.anomaly_roi_polygon = np.array(vid_pts, dtype=np.int32)
        n = len(canvas_pts)
        self.anomaly_zone_status.config(
            text=f"● Custom zone  ({n} points)", fg=ACCENT_CYAN)
        self._anomaly_exit_draw_mode()
        self.anomaly_pts_lbl.config(text=f"Zone set: {n} pts — ready")
        # Redraw final polygon (non-interactive)
        self._anomaly_redraw_zone()

    def _anomaly_enter_draw_mode(self):
        """Enter interactive zone-draw mode."""
        if self._anomaly_first_frame is None:
            messagebox.showwarning(
                "No Frame",
                "Load a video or start webcam first so a preview frame is available.")
            return
        if self.anomaly_is_running:
            messagebox.showinfo("Running", "Stop detection before redrawing the zone.")
            return
        self.anomaly_draw_mode   = True
        self.anomaly_zone_points = []
        self.anomaly_roi_polygon = None
        self._anomaly_mouse_pos  = (0, 0)
        # Show the first frame as background
        self._anomaly_canvas_render_frame(self._anomaly_first_frame)
        self._anomaly_redraw_zone()
        # Show instruction bar (pack after the header divider line at top)
        self.anomaly_instruction_bar.pack(fill="x", padx=25, pady=(0, 4))
        self.anomaly_zone_status.config(
            text="● Drawing…  click to add points", fg=ACCENT_AMBER)
        self.anomaly_pts_lbl.config(text="Points: 0  |  ENTER to confirm  |  C to clear")
        self.anomaly_canvas.focus_set()
        self.anomaly_draw_zone_btn.config(text="✔  CONFIRM ZONE",
                                           bg=ACCENT_GREEN, fg=BG_DARK,
                                           command=self._anomaly_canvas_confirm)
        self.anomaly_canvas.focus_set()

    def _anomaly_exit_draw_mode(self):
        self.anomaly_draw_mode = False
        try:
            self.anomaly_instruction_bar.pack_forget()
        except Exception:
            pass
        self.anomaly_draw_zone_btn.config(text="✏  DRAW ZONE",
                                           bg=ACCENT_CYAN, fg=BG_DARK,
                                           command=self._anomaly_enter_draw_mode)

    # ── Source loaders ──────────────────────────────────────────────────
    def _anomaly_load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if not path:
            return
        self.anomaly_video_path = path
        name = os.path.basename(path)
        self.anomaly_source_lbl.config(text=name, fg=ACCENT_CYAN)
        self.anomaly_phase_lbl.config(
            text=f"[ LOADED: {name} ]", fg=ACCENT_CYAN)
        # Extract first frame for zone preview
        cap_p = cv2.VideoCapture(path)
        ret_p, frame_p = cap_p.read()
        cap_p.release()
        if ret_p:
            self._anomaly_first_frame = cv2.resize(frame_p, (1280, 720))
            self.after(50, self._anomaly_show_preview)
        self.anomaly_zone_status.config(
            text="● Full frame (default)", fg=ACCENT_GREEN)
        self.anomaly_roi_polygon = None
        self.anomaly_zone_points = []
        self.anomaly_pts_lbl.config(text="Draw a zone or keep full-frame default")

    def _anomaly_use_webcam(self):
        self.anomaly_video_path = 0
        self.anomaly_source_lbl.config(text="Webcam (device 0)", fg=ACCENT_CYAN)
        self.anomaly_phase_lbl.config(text="[ WEBCAM READY ]", fg=ACCENT_CYAN)
        cap_w = cv2.VideoCapture(0)
        ret_w, frame_w = cap_w.read()
        cap_w.release()
        if ret_w:
            self._anomaly_first_frame = cv2.resize(frame_w, (1280, 720))
            self.after(50, self._anomaly_show_preview)
        self.anomaly_zone_status.config(text="● Full frame (default)", fg=ACCENT_GREEN)
        self.anomaly_roi_polygon = None
        self.anomaly_zone_points = []
        self.anomaly_pts_lbl.config(text="Draw a zone or keep full-frame default")

    def _anomaly_show_preview(self):
        """Show the first frame on the canvas as a static preview."""
        if self._anomaly_first_frame is None:
            return
        self._anomaly_canvas_render_frame(self._anomaly_first_frame)
        # Re-draw any existing zone
        self._anomaly_redraw_zone()

    def _anomaly_clear_zone(self):
        self.anomaly_zone_points = []
        self.anomaly_roi_polygon = None
        self.anomaly_draw_mode   = False
        self._anomaly_exit_draw_mode()
        self.anomaly_zone_status.config(text="● Full frame (default)", fg=ACCENT_GREEN)
        self.anomaly_pts_lbl.config(text="")
        if self._anomaly_first_frame is not None:
            self._anomaly_show_preview()

    # ── Detection controls ──────────────────────────────────────────────
    def _anomaly_start(self):
        if self.anomaly_is_running:
            return
        if self.anomaly_video_path is None:
            messagebox.showwarning(
                "No Source", "Load a video file or select webcam first.")
            return
        # Exit draw mode if still active
        if self.anomaly_draw_mode:
            self._anomaly_canvas_escape()

        self.anomaly_is_running = True
        self.anomaly_results    = []
        for lbl in self.anomaly_counters.values():
            lbl.config(text="0")
        self.anomaly_prog["value"] = 0
        self.anomaly_prog_lbl.config(text="0%", fg=TEXT_DIM)
        self.anomaly_start_btn.config(state="disabled", bg=TEXT_DIM)
        self.anomaly_stop_btn.config(state="normal")
        self.anomaly_phase_lbl.config(text="[ INITIALISING… ]", fg=ACCENT_AMBER)
        self._set_sidebar_status("searching", "Anomaly detection active")
        threading.Thread(target=self._anomaly_loop,
                         args=(self.anomaly_video_path,), daemon=True).start()

    def _anomaly_stop(self):
        self.anomaly_is_running = False
        self.anomaly_start_btn.config(state="normal", bg=ACCENT_GREEN)
        self.anomaly_stop_btn.config(state="disabled")
        self.anomaly_phase_lbl.config(text="[ STOPPED ]", fg=ACCENT_RED)
        self._set_sidebar_status("idle", "Anomaly detection stopped")

    # ── Detection loop (background thread) ─────────────────────────────
    def _anomaly_loop(self, source):
        try:
            import anomaly as _am
            from anomaly import (select_device, CONF_THRESHOLD, SAVE_DIR,
                                  FALL_SAVE_DIR, PanicCrowdDetector,
                                  FallFightDetector, LoiteringDetector,
                                  CSVLogger, UI)
            from ultralytics import YOLO
            import numpy as np
        except ImportError as e:
            self.after(0, lambda err=str(e): messagebox.showerror(
                "Import Error", f"Cannot load anomaly module:\n{err}"))
            self.after(0, self._anomaly_stop)
            return

        try:
            device   = select_device()
            model    = YOLO(_am.MODEL_PATH)
            use_half = (device == "cuda")
            logger   = CSVLogger(self.anomaly_log_path)

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.after(0, lambda: messagebox.showerror(
                    "Error", "Cannot open video source."))
                self.after(0, self._anomaly_stop)
                return

            src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
            fps          = max(src_fps, 1.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            W, H         = 1280, 720

            # Use custom ROI or full-frame default
            if self.anomaly_roi_polygon is not None:
                roi = self.anomaly_roi_polygon
            else:
                roi = np.array([[0,0],[W,0],[W,H],[0,H]], np.int32)

            os.makedirs(SAVE_DIR,      exist_ok=True)
            os.makedirs(FALL_SAVE_DIR, exist_ok=True)

            panic_det  = PanicCrowdDetector(fps, logger, roi, SAVE_DIR, W, H)
            fall_det   = FallFightDetector(fps, logger, FALL_SAVE_DIR)
            loiter_det = LoiteringDetector(fps, logger, roi, SAVE_DIR, W, H)

            event_counts  = {k: 0 for k in self.anomaly_counters}
            _alert_active = {}   # tid -> set of currently-active alert types
            frame_count   = 0

            self.after(0, lambda: self.anomaly_phase_lbl.config(
                text="[ SCANNING… ]", fg=ACCENT_AMBER))

            while cap.isOpened() and self.anomaly_is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                frame = cv2.resize(frame, (W, H))

                # Draw ROI boundary on frame
                pts = roi.reshape(-1, 2)
                n   = len(pts)
                for k in range(n):
                    p1 = tuple(pts[k])
                    p2 = tuple(pts[(k+1) % n])
                    cv2.line(frame, p1, p2, (0, 229, 255), 1, cv2.LINE_AA)

                active_alerts = []
                results = model.track(
                    frame, persist=True, conf=CONF_THRESHOLD,
                    verbose=False, device=device, half=use_half, iou=0.5)

                boxes = keypoints_xy = keypoints_xyn = None
                current_ids = []
                if (results[0].boxes is not None and
                        results[0].boxes.id is not None):
                    boxes         = results[0].boxes.data.cpu().numpy()
                    keypoints_xy  = results[0].keypoints.data.cpu().numpy()
                    keypoints_xyn = results[0].keypoints.xyn.cpu().numpy()
                    current_ids   = [int(b[4]) for b in boxes]

                panic_state, crowd_members = panic_det.compute(
                    frame, boxes, keypoints_xy, keypoints_xyn,
                    frame_count, active_alerts=active_alerts)
                fall_fight_data = fall_det.compute(
                    frame, boxes, keypoints_xy, keypoints_xyn,
                    frame_count, active_alerts=active_alerts,
                    all_box_count=len(current_ids))
                loiter_data = loiter_det.compute(
                    frame, boxes, keypoints_xy,
                    frame_count, active_alerts=active_alerts)

                for mod in (panic_det, fall_det, loiter_det):
                    mod.cleanup_stale(current_ids, frame_count)

                # Count and draw
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        tid = int(box[4])
                        x1,y1,x2,y2 = map(int, box[:4])
                        p_s  = panic_state.get(tid, {})
                        ff_s = fall_fight_data.get(tid, {})
                        lo_s = loiter_data.get(tid, {})

                        has_fall  = ff_s.get("has_fall",     False)
                        in_crowd  = p_s.get("is_in_crowd",  False)
                        loiter    = lo_s.get("is_alert",     False)
                        running   = p_s.get("is_running",   False)
                        any_alert = has_fall or in_crowd or loiter or running

                        # Count each alert only on rising edge (False → True)
                        prev = _alert_active.get(tid, set())
                        cur  = set()
                        if has_fall  and "FALL"      not in prev: event_counts["FALL"]      += 1
                        if in_crowd  and "CROWD"     not in prev: event_counts["CROWD"]     += 1
                        if loiter    and "LOITERING" not in prev: event_counts["LOITERING"] += 1
                        if running   and "RUNNING"   not in prev: event_counts["RUNNING"]   += 1
                        if has_fall:  cur.add("FALL")
                        if in_crowd:  cur.add("CROWD")
                        if loiter:    cur.add("LOITERING")
                        if running:   cur.add("RUNNING")
                        _alert_active[tid] = cur

                        color = (UI.FALL  if has_fall  else
                                 UI.CROWD if in_crowd  else
                                 UI.ALERT if (loiter or running) else UI.WATCHING)
                        UI.corner_mark(frame, x1,y1,x2,y2, color,
                                       length=16 if any_alert else 10,
                                       thickness=2 if any_alert else 1)
                        if has_fall:
                            UI.alert_badge(frame, "FALL DETECTED",   x1, y1-10, UI.FALL)
                        elif in_crowd:
                            UI.alert_badge(frame, "CROWD ALERT",     x1, y1-10, UI.CROWD)
                        elif loiter:
                            UI.alert_badge(frame, "LOITERING",       x1, y1-10, UI.ALERT)
                        elif running:
                            UI.alert_badge(frame, "RUNNING ALERT",   x1, y1-10, UI.ALERT)

                UI.draw_hud(frame, frame_count, fps, active_alerts, device=device)

                pct    = min(int((frame_count / total_frames) * 99), 99)
                ts_str = f"Frame {frame_count} / {total_frames}"

                # Push frame to canvas on main thread
                self.after(0, lambda f=frame.copy(), t=ts_str:
                           self._anomaly_push_frame(f, t))
                self.after(0, lambda p=pct: self._anomaly_update_progress(p))

                if frame_count % 30 == 0:
                    self.after(0, lambda ec=dict(event_counts):
                               self._anomaly_update_counters(ec))
                if frame_count % 60 == 0:
                    self.after(0, self._anomaly_refresh_log)

            cap.release()
            for mod in (panic_det, fall_det, loiter_det):
                mod.release()
            logger.shutdown()

            total_ev = sum(event_counts.values())
            self.after(0, lambda: self._anomaly_update_progress(100))
            self.after(0, lambda: self.anomaly_phase_lbl.config(
                text=f"[ COMPLETE  —  {total_ev} events ]", fg=ACCENT_GREEN))
            self.after(0, lambda: self._set_sidebar_status(
                "done", f"Anomaly done — {total_ev} events"))
            self.after(0, lambda ec=dict(event_counts):
                       self._anomaly_update_counters(ec))
            self.after(0, self._anomaly_refresh_log)
            if hasattr(self, "_tab_count_labels") and "anomaly" in self._tab_count_labels:
                self.after(0, lambda: self._tab_count_labels["anomaly"].config(
                    text=str(len(self._scan_anomaly_results()))))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda err=str(e): messagebox.showerror(
                "Anomaly Error", f"Detection failed:\n{err}"))
            print(tb)
        finally:
            self.after(0, lambda: self.anomaly_start_btn.config(
                state="normal", bg=ACCENT_GREEN))
            self.after(0, lambda: self.anomaly_stop_btn.config(state="disabled"))
            self.anomaly_is_running = False

    # ── Frame push (main thread) ────────────────────────────────────────
    def _anomaly_push_frame(self, bgr_frame, ts_str):
        """Render a detection frame onto the canvas (not draw mode)."""
        if self.anomaly_draw_mode:
            return
        cw = self.anomaly_canvas.winfo_width()  or 800
        ch = self.anomaly_canvas.winfo_height() or 450
        img = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        img = img.resize((cw, ch), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.anomaly_canvas.delete("all")
        self.anomaly_canvas.create_image(0, 0, anchor="nw", image=img_tk, tags="frame")
        self._anomaly_img_tk = img_tk
        self.anomaly_frame_lbl.config(text=ts_str)

    # ── Misc updates ────────────────────────────────────────────────────
    def _anomaly_update_progress(self, value):
        self.anomaly_prog["value"] = value
        color = ACCENT_CYAN if value < 100 else ACCENT_GREEN
        self.anomaly_prog_lbl.config(text=f"{value}%", fg=color)

    def _anomaly_update_counters(self, counts):
        for key, lbl in self.anomaly_counters.items():
            lbl.config(text=str(counts.get(key, 0)))

    def _anomaly_refresh_log(self):
        """Reload security_logs.csv and display newest entries."""
        import csv as _csv
        path = self.anomaly_log_path
        if not os.path.isfile(path):
            return
        try:
            rows = []
            with open(path, newline="") as f:
                rows = list(_csv.reader(f))
            display = rows[1:][-60:]   # skip header, last 60
            self.anomaly_log_listbox.delete(0, "end")
            if not display:
                self.anomaly_log_listbox.insert("end", "  No events logged yet.")
                return
            for row in reversed(display):
                if len(row) >= 4:
                    ts, tid, event, status = row[0], row[1], row[2], row[3]
                    self.anomaly_log_listbox.insert(
                        "end", f"  {ts}  ID:{tid}  {event}  [{status}]")
                    idx = self.anomaly_log_listbox.size() - 1
                    ev = event.upper()
                    fg = ("#ff4757" if "FALL"   in ev else
                          "#ffa502" if "CROWD"  in ev else
                          "#00e5ff" if "LOITER" in ev else
                          "#ffa502" if "RUN"    in ev else
                          "#7a8fa8")
                    self.anomaly_log_listbox.itemconfig(idx, fg=fg)
        except Exception:
            pass

    def show_page(self, name):
        self.is_running = False # Gracefully stop active threads
        
        # Hide all frames
        for page in [self.home_page, self.reid_page, self.analytics_page,
                     self.zone_page, self.results_page, self.suspect_page,
                     self.anomaly_page]:
            page.pack_forget()

        self._set_active_nav(name)

        # Show selected frame
        if name == "Home": 
            self.home_page.pack(fill="both", expand=True)
        elif name == "ReID": 
            self.reid_page.pack(fill="both", expand=True)
        elif name == "Analytics": 
            self.analytics_page.pack(fill="both", expand=True)
        elif name == "Zones": 
            self.zone_page.pack(fill="both", expand=True)
        elif name == "Results": 
            self.results_page.pack(fill="both", expand=True)
            self._refresh_results_page()
        elif name == "Suspect":
            self.suspect_page.pack(fill="both", expand=True)
        elif name == "Anomaly":
            self.anomaly_page.pack(fill="both", expand=True)
            self._anomaly_refresh_log()
    # ------------------------------------------------------------------ #
    #  CLICK HANDLING                                                      #
    # ------------------------------------------------------------------ #
    def handle_click(self, event):
        if not self.latest_detections:
            return

        label_w = self.vid_label.winfo_width()
        label_h = self.vid_label.winfo_height()
        orig_w, orig_h = self.current_frame_size
        real_x = event.x * (orig_w / label_w)
        real_y = event.y * (orig_h / label_h)

        for (xyxy, tid) in self.latest_detections:
            x1, y1, x2, y2 = xyxy
            if x1 <= real_x <= x2 and y1 <= real_y <= y2:
                self.selected_tid = tid
                self.engine.target_gallery = []
                self.locked_id_lbl.config(text=f"#{tid}", fg=ACCENT_AMBER)
                self._set_status(f"Locked on ID #{tid} — collecting features...", ACCENT_AMBER)
                self._set_sidebar_status("active", f"Collecting features for ID #{tid}")
                break

    # ------------------------------------------------------------------ #
    #  PHASE 1 – SELECTION                                                 #
    # ------------------------------------------------------------------ #
    def start_p1(self):
        # Complete local state reset
        self.is_running = True
        self.search_complete = False
        self.match_results = []
        self.match_queue = []
        self.selected_tid = None
        self.latest_detections = []
        self.current_result_idx = 0
        
        # Full engine reset before starting
        self.engine.full_reset()
        
        # Reset UI Elements
        self.locked_id_lbl.config(text="—")
        self.phase_lbl.config(text="[ PHASE 1: TARGET SELECTION ]", fg=ACCENT_AMBER)
        self.start_btn.config(state="disabled", bg=TEXT_DIM)
        self.stop_reid_btn.config(state="normal")
        
        # Safe config for labels that might have different names
        if hasattr(self, 'live_match_lbl'): self.live_match_lbl.config(text="0")
        if hasattr(self, 'match_count_lbl'): self.match_count_lbl.config(text="0")
        
        # Reset progress bar explicitly
        self.prog_bar.config(value=0)
        self.prog_pct_lbl.config(text="0%", fg=TEXT_DIM)
        
        self._set_status("Initializing camera... please click target person.")
        self._set_sidebar_status("active", "Phase 1: Selecting Target")
        
        # Small delay to ensure UI updates before thread starts
        self.after(100, lambda: threading.Thread(target=self.run_p1_loop, daemon=True).start())

    def run_p1_loop(self):
        cap = cv2.VideoCapture(VIDEO_1)
        self.current_frame_size = (int(cap.get(3)), int(cap.get(4)))

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            if len(self.engine.target_gallery) >= 15:
                break

            processed, detections = self.engine.process_frame(frame, self.selected_tid)
            self.latest_detections = detections

            if self.selected_tid is not None:
                for xyxy, tid in detections:
                    if tid == self.selected_tid:
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = frame[y1:y2, x1:x2]
                        feat = self.engine.get_features(crop)
                        if feat is not None:
                            self.engine.target_gallery.append(feat)

            # Use centralized image helper to keep sizing consistent
            self.after(0, self._apply_image_to_label, processed, self.vid_label)

        cap.release()
        if len(self.engine.target_gallery) >= 15:
            self.after(0, self.start_p2_background)
        else:
            self.after(0, lambda: self._set_status(
                "Phase 1 ended — target not fully profiled. Try again.", ACCENT_RED))
            self.after(0, lambda: self.start_btn.config(state="normal", bg=ACCENT_GREEN))
            self.after(0, lambda: self.stop_reid_btn.config(state="disabled"))

    # ------------------------------------------------------------------ #
    #  PHASE 2 – BACKGROUND SEARCH                                         #
    # ------------------------------------------------------------------ #
    def start_p2_background(self):
        # Reset progress bar explicitly before Phase 2 starts
        self.prog_bar.config(value=0)
        self.prog_pct_lbl.config(text="0%", fg=ACCENT_CYAN)
        
        self.phase_lbl.config(text="[ PHASE 2: SEARCHING FOOTAGE ]", fg=ACCENT_CYAN)
        self._set_status("Searching secondary footage...", ACCENT_CYAN)
        self._set_sidebar_status("searching", "Phase 2: Scanning footage")
        self.stop_reid_btn.config(state="normal")  # Ensure stop is enabled
        
        gallery_array = np.array(self.engine.target_gallery)
        threading.Thread(
            target=self.engine.search_video,
            args=(VIDEO_2, gallery_array, self.update_progress,
                  self.on_match_found, lambda: not self.is_running, MATCHES_DIR),
            daemon=True
        ).start()
        self.check_match_queue()

    def update_progress(self, value):
        # Called from background thread — must route ALL Tk updates through after()
        self.after(0, self._apply_progress, value)

    def _apply_progress(self, value):
        """Runs on the main Tk thread."""
        self.prog_bar["value"] = value
        self.prog_pct_lbl.config(text=f"{value}%",
                                  fg=ACCENT_CYAN if value < 100 else ACCENT_GREEN)
        if value >= 100:
            self.search_complete = True
            # Give the UI queue 800ms to drain all remaining match callbacks
            self.after(800, self._on_search_complete)

    def _on_search_complete(self):
        total = len(self.match_results)
        self.phase_lbl.config(text="[ SEARCH COMPLETE ]", fg=ACCENT_GREEN)
        self._set_status(f"Search complete — {total} match(es) found", ACCENT_GREEN)
        self._set_sidebar_status("done", f"Done — {total} matches")
        self.status_dot.config(text="● DONE", fg=ACCENT_GREEN)
        self.match_count_lbl.config(text=str(total))
        self.live_match_lbl.config(text=str(total))
        self.results_total_lbl.config(text=str(total))
        self.start_btn.config(state="normal", bg=ACCENT_GREEN)
        self.stop_reid_btn.config(state="disabled")

        if total > 0:
            # Show View Results button on reid page (only if it exists)
            if hasattr(self, 'reid_results_btn'):
                self.reid_results_btn.pack(fill="x", pady=(10, 0))
            # Show in sidebar
            self.view_results_btn.pack(fill="x", padx=10, pady=8)

    def on_match_found(self, filepath, video_name, timestamp):
        # Called from background thread — route to main thread via after()
        self.after(0, self.match_queue.append, (filepath, video_name, timestamp))

    def check_match_queue(self):
        # Only process if we're actively searching or waiting for search completion
        if not self.is_running and not self.search_complete:
            return
        
        while self.match_queue:
            filepath, video_name, ts = self.match_queue.pop(0)
            self.match_results.append((filepath, video_name, ts))
            count = len(self.match_results)
            self.live_match_lbl.config(text=str(count))
            self.match_count_lbl.config(text=str(count))

        # Keep polling until search is done AND queue is fully empty
        if (self.is_running or self.search_complete) and (not self.search_complete or self.match_queue):
            self.after(150, self.check_match_queue)

    # ------------------------------------------------------------------ #
    #  RESULTS VIEWER                                                      #
    # ------------------------------------------------------------------ #
    def _refresh_results_page(self):
        """Called when navigating to the Results page — refresh active tab."""
        if not hasattr(self, '_active_results_tab'):
            self._active_results_tab = "reid"
        if not hasattr(self, '_active_items'):
            self._active_items = []
        # Update all tab counts
        self._tab_count_labels["reid"].config(text=str(len(self._scan_reid_results())))
        self._tab_count_labels["zones"].config(text=str(len(self._scan_zone_results())))
        self._tab_count_labels["heatmaps"].config(text=str(len(self._scan_heatmap_results())))
        self._tab_count_labels["suspect"].config(text=str(len(self._scan_suspect_results())))
        self._tab_count_labels["anomaly"].config(text=str(len(self._scan_anomaly_results())))
        # Load the currently active tab
        self._load_tab_items(self._active_results_tab)

    def _show_active_item(self, idx):
        """Display item at idx from self._active_items."""
        items = self._active_items
        if not items:
            return
        idx = max(0, min(idx, len(items) - 1))
        self.current_result_idx = idx
        filepath, source_label, timestamp = items[idx]
        try:
            img = Image.open(filepath)
            self.result_original_image = img.copy()
            self.result_zoom_level = 1.0
            self._update_result_display()
        except Exception as e:
            self.result_img_lbl.config(image="", text=f"[ Error: {e} ]")
        total = len(items)
        self.meta_fields["source"].config(text=source_label, wraplength=240)
        self.meta_fields["timestamp"].config(text=timestamp, fg=ACCENT_CYAN)
        self.meta_fields["index"].config(text=f"#{idx + 1} of {total}")
        self.meta_fields["saved"].config(text=os.path.basename(filepath), fg=ACCENT_GREEN)
        self.nav_pos_lbl.config(text=f"{idx + 1} / {total}")
        self.prev_btn.config(fg=TEXT_PRIMARY if idx > 0 else TEXT_DIM,
                              state="normal" if idx > 0 else "disabled")
        self.next_btn.config(fg=TEXT_PRIMARY if idx < total - 1 else TEXT_DIM,
                              state="normal" if idx < total - 1 else "disabled")

    def _show_result(self, idx):
        """Backwards-compatible alias."""
        self._show_active_item(idx)

    def _update_result_display(self):
        """Render the image with current zoom level."""
        if self.result_original_image is None:
            return
        
        img = self.result_original_image.copy()
        
        # Base display dimensions
        base_w, base_h = 580, 480
        
        # Apply zoom (scale dimensions)
        zoom_w = int(base_w * self.result_zoom_level)
        zoom_h = int(base_h * self.result_zoom_level)
        
        # Resize image
        img.thumbnail((zoom_w, zoom_h), Image.LANCZOS)
        
        # Add padding to center (always fit to display area)
        bg = Image.new("RGB", (base_w, base_h), (4, 8, 16))
        offset_x = (base_w - img.width) // 2
        offset_y = (base_h - img.height) // 2
        bg.paste(img, (offset_x, offset_y))
        
        img_tk = ImageTk.PhotoImage(bg)
        self.result_img_lbl.config(image=img_tk, text="")
        self.result_img_lbl._img = img_tk
        
        # Update zoom display
        zoom_pct = int(self.result_zoom_level * 100)
        self.zoom_lbl.config(text=f"{zoom_pct}%")

    def zoom_in_result(self):
        """Increase zoom level."""
        if self.result_original_image is None:
            return
        self.result_zoom_level = min(self.result_zoom_level + 0.2, 3.0)  # Max 300%
        self._update_result_display()

    def zoom_out_result(self):
        """Decrease zoom level."""
        if self.result_original_image is None:
            return
        self.result_zoom_level = max(self.result_zoom_level - 0.2, 0.5)  # Min 50%
        self._update_result_display()

    def reset_zoom(self):
        """Reset zoom to 100% fit."""
        if self.result_original_image is None:
            return
        self.result_zoom_level = 1.0
        self._update_result_display()

    def _on_results_mousewheel(self, event):
        """Handle mouse wheel zoom on results image."""
        # event.delta > 0 = scroll up = zoom in (Windows)
        # event.num == 4 = scroll up (Linux)
        # event.num == 5 = scroll down (Linux)
        if event.delta > 0 or event.num == 4:
            self.zoom_in_result()
        else:
            self.zoom_out_result()

    def show_prev_result(self):
        self._show_result(self.current_result_idx - 1)
        self.result_img_lbl.focus_set()

    def show_next_result(self):
        self._show_result(self.current_result_idx + 1)
        self.result_img_lbl.focus_set()

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #
    def select_zone_video(self):
        """File picker for the Zone Monitoring module."""
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if path: 
            self._set_sidebar_status("active", "Zone Setup: Ready")
            self.start_zone_thread(path)

    def start_zone_thread(self, source):
        """Initializes monitoring and launches the background thread."""
        self.is_running = True
        self._set_sidebar_status("searching", "Monitoring Zones...")
        threading.Thread(target=self.run_zone_monitoring, args=(source,), daemon=True).start()
    
    def activate_zone_ai(self):
        """Activates AI monitoring mode after zones have been drawn."""
        if not self.zone_engine.zones:
            messagebox.showwarning("Zone Setup", "Please draw at least one zone first!")
            return
        self.zone_ai_active = True
        self.start_ai_btn.config(state="disabled", text="▶ AI MONITORING ACTIVE", fg=ACCENT_AMBER)
        self._set_sidebar_status("searching", "AI Monitoring Active - Processing...")
        self._set_status("AI zone monitoring activated. Video playback started.", ACCENT_GREEN)
    
    def record_zone_point(self, event):
        """Handle left-click to add polygon point."""
        if self.is_running and hasattr(self, 'setup_frame') and self.setup_frame is not None:
            # Convert screen coords to video coords
            label_width = self.zone_vid_label.winfo_width()
            label_height = self.zone_vid_label.winfo_height()
            video_width = self.setup_frame.shape[1]
            video_height = self.setup_frame.shape[0]
            
            if label_width < 10 or label_height < 10:
                return
            
            # Pixel-to-video coordinate mapping
            x_ratio = video_width / label_width
            y_ratio = video_height / label_height
            video_x = int(event.x * x_ratio)
            video_y = int(event.y * y_ratio)
            
            # Clamp to video bounds
            video_x = max(0, min(video_x, video_width - 1))
            video_y = max(0, min(video_y, video_height - 1))
            
            self.temp_points.append((video_x, video_y))
            
            # Show preview with current points
            preview = self.zone_engine.draw_preview(self.setup_frame.copy(), self.temp_points)
            self.after(0, lambda: self._apply_image_to_label(preview, self.zone_vid_label))
            self._set_status(f"Point added: {len(self.temp_points)} points", ACCENT_CYAN)
    
    def finalize_zone(self, event):
        """Handle right-click to finalize polygon."""
        if len(self.temp_points) < 3:
            messagebox.showwarning("Zone Drawing", "Need at least 3 points to create a zone!")
            self.temp_points = []
            return
        
        # Add zone to engine
        self.zone_engine.add_zone(self.temp_points)
        self.temp_points = []
        
        # Enable AI button and show confirmation
        self.start_ai_btn.config(state="normal")
        self._set_status(f"Zone added! Total zones: {len(self.zone_engine.zones)}", ACCENT_GREEN)
        
        # Show preview with new zone
        if hasattr(self, 'setup_frame') and self.setup_frame is not None:
            preview = self.zone_engine.draw_preview(self.setup_frame.copy(), [])
            self.after(0, lambda: self._apply_image_to_label(preview, self.zone_vid_label))
    
    def run_zone_monitoring(self, source):
        """Main zone monitoring loop - freezes on first frame until AI monitoring is activated."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror("Error", "Cannot open video source!"))
            self.is_running = False
            return
        
        # Get first frame for drawing reference
        ret, frame = cap.read()
        if not ret:
            cap.release()
            self.is_running = False
            return
        
        self.setup_frame = frame.copy()
        self.video_cap = cap  # Store for access from activate_zone_ai
        self.after(0, lambda: self._apply_image_to_label(frame, self.zone_vid_label))
        self._set_sidebar_status("active", "Zone Drawing Mode - Ready for Zones")
        
        # Get frame rate for proper playback timing
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1000 // fps  # Convert to milliseconds
        last_time = time.time()
        
        while self.is_running and cap.isOpened():
            # If AI monitoring is active, process zone detection
            if self.zone_ai_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Control frame rate
                elapsed = (time.time() - last_time) * 1000
                if elapsed < frame_delay:
                    time.sleep((frame_delay - elapsed) / 1000)
                last_time = time.time()
                
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                result_frame, alerts = self.zone_engine.process_frame(frame, timestamp_str, ZONE_ALERTS_DIR)
                
                # Display results and alerts
                self.after(0, lambda f=result_frame: self._apply_image_to_label(f, self.zone_vid_label))
                
                # Add alerts to the list
                for alert in alerts:
                    self.after(0, lambda a=alert: self.alert_box.insert(0, a))
            else:
                # Freeze on first frame - wait for user to activate monitoring
                time.sleep(0.1)
        
        cap.release()
        self.video_cap = None
        self.is_running = False
        self.zone_ai_active = False
        self.start_ai_btn.config(state="normal", text="▶ START AI MONITORING", fg=TEXT_PRIMARY)
        self.after(0, lambda: self._set_sidebar_status("idle", "Monitoring Complete"))

    # Zone methods above (activate_zone_ai, record_zone_point, finalize_zone, run_zone_monitoring, stop_processing)

    def update_display(self, frame):
        # Called from Phase 1 thread — schedule on main thread
        self.after(0, self._apply_display, frame.copy())

    def _apply_display(self, frame):
        lw = self.vid_label.winfo_width()
        lh = self.vid_label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = 800, 450
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((lw, lh))
        img_tk = ImageTk.PhotoImage(img)
        self.vid_label.config(image=img_tk, text="")
        self.vid_label._img_tk = img_tk

    def _set_status(self, text, color=TEXT_SECONDARY):
        self.status_lbl.config(text=text, fg=color)

    def _set_sidebar_status(self, mode, text):
        colors = {"active": ACCENT_AMBER, "searching": ACCENT_CYAN,
                  "done": ACCENT_GREEN, "idle": TEXT_DIM}
        icons  = {"active": "● ACTIVE", "searching": "◌ SCANNING",
                  "done": "✔ COMPLETE", "idle": "● IDLE"}
        self.status_dot.config(text=icons.get(mode, "●"), fg=colors.get(mode, TEXT_DIM))
        self.sidebar_status.config(text=text)

    def stop_processing(self):
        """Global stop for all active AI video threads."""
        self.is_running = False
        self._set_sidebar_status("idle", "System Standby")
        self._set_status("All operations stopped.", TEXT_SECONDARY)
        messagebox.showinfo("System Info", "Processing stopped manually.")

    def stop_reid(self):
        """Stop ReID and perform a deep reset of UI and Engine."""
        # STEP 1: Stop the thread by setting is_running to False (this triggers stop_check() in search_video)
        self.is_running = False
        self.search_complete = False
        
        # STEP 2: Reset the AI Engine completely
        self.engine.full_reset()
        
        # STEP 3: Clear all UI state variables
        self.selected_tid = None
        self.latest_detections = []
        self.match_queue = []
        self.match_results = []
        self.current_result_idx = 0
        
        # STEP 4: Clear the Result Gallery if it exists (thumbnails at bottom)
        if hasattr(self, 'inner_gal'):
            for widget in self.inner_gal.winfo_children():
                widget.destroy()
        
        # STEP 5: Reset all UI Labels & Progress
        self.locked_id_lbl.config(text="—")
        self.live_match_lbl.config(text="0")
        self.match_count_lbl.config(text="0")
        self.prog_bar.config(value=0)
        self.prog_pct_lbl.config(text="0%", fg=TEXT_DIM)
        self.phase_lbl.config(text="[ SYSTEM RESET ]", fg=ACCENT_RED)
        self.vid_label.config(image="", text="[ VIDEO FEED RESET ]", fg=TEXT_DIM)
        
        # STEP 6: Update button states
        self.start_btn.config(state="normal", bg=ACCENT_GREEN)
        self.stop_reid_btn.config(state="disabled")
        
        # STEP 7: Update status messages
        self._set_status("System fully reset. Ready for Phase 1.", TEXT_SECONDARY)
        self._set_sidebar_status("idle", "Ready to begin")

    def _build_zone_page(self):
        p = self.zone_page # Make sure to initialize self.zone_page in __init__
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="VIRTUAL RESTRICTED ZONES", font=self.font_head, fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        
        main_cols = tk.Frame(p, bg=BG_DARK)
        main_cols.pack(fill="both", expand=True, padx=25, pady=15)

        # --- RIGHT SIDEBAR (ALERTS) ---
        right_col = tk.Frame(main_cols, bg=BG_PANEL, width=350)
        right_col.pack(side="right", fill="y", padx=(15, 0))
        right_col.pack_propagate(False)

        tk.Label(right_col, text="INTRUSION ALERTS", font=self.font_mono, fg=ACCENT_RED, bg=BG_PANEL).pack(pady=10)

        # Monitoring controls (start/stop source)
        tk.Label(right_col, text="MONITORING CONTROLS", font=self.font_head, fg=ACCENT_CYAN, bg=BG_PANEL).pack(pady=10)
        tk.Button(right_col, text="📁 LOAD VIDEO", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY,
              command=lambda: self.select_zone_video()).pack(fill="x", padx=20, pady=5)
        tk.Button(right_col, text="📷 LIVE WEBCAM", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY,
              command=lambda: self.start_zone_thread(0)).pack(fill="x", padx=20, pady=5)

        self.start_ai_btn = tk.Button(right_col, text="▶ START AI MONITORING", font=self.font_btn, 
                                      bg=ACCENT_GREEN, fg=TEXT_PRIMARY, state="disabled", 
                                      command=self.activate_zone_ai, pady=12)
        self.start_ai_btn.pack(fill="x", padx=20, pady=10)

        # Scrollable Alert List
        self.alert_box = tk.Listbox(right_col, bg=BG_DARK, fg=ACCENT_RED, font=self.font_mono, 
                         borderwidth=0, highlightthickness=1, highlightbackground=BORDER)
        self.alert_box.pack(fill="both", expand=True, padx=10, pady=5)

        tk.Button(right_col, text="CLEAR ZONES", font=self.font_btn, bg=BG_CARD, fg=TEXT_PRIMARY, 
                  command=self.zone_engine.clear_zones).pack(fill="x", padx=20, pady=5)
        
        self.stop_zone_btn = tk.Button(right_col, text="■ STOP MONITORING", font=self.font_btn, 
                                       bg=ACCENT_RED, fg=TEXT_PRIMARY, pady=12, command=self.stop_processing)
        self.stop_zone_btn.pack(fill="x", padx=20, pady=10)

        # --- LEFT AREA (VIDEO) ---
        left_col = tk.Frame(main_cols, bg=BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        feed_wrap = tk.Frame(left_col, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_BRIGHT)
        feed_wrap.pack(fill="both", expand=True)
        
        self.zone_vid_label = tk.Label(feed_wrap, bg="#040810", text="[ CLICK TO DRAW POLYGON ]", font=self.font_sub, fg=TEXT_DIM)
        self.zone_vid_label.pack(fill="both", expand=True, padx=4, pady=4)
        
        # Drawing Bindings
        self.zone_vid_label.bind("<Button-1>", self.record_zone_point)
        self.zone_vid_label.bind("<Button-3>", self.finalize_zone)
        
        self.temp_points = []
        self.zone_ai_active = False  # Toggle between drawing and monitoring modes


    # ================================================================== #
    #  SUSPECT FINDER PAGE                                                #
    # ================================================================== #
    def _build_suspect_page(self):
        p = self.suspect_page

        # Header
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=25, pady=(20, 10))
        tk.Label(hdr, text="SUSPECT FINDER", font=("Courier New", 16, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")
        self.suspect_phase_lbl = tk.Label(hdr, text="[ IDLE ]",
                                          font=("Courier New", 10, "bold"),
                                          fg=TEXT_DIM, bg=BG_DARK)
        self.suspect_phase_lbl.pack(side="right")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=25)

        cols = tk.Frame(p, bg=BG_DARK)
        cols.pack(fill="both", expand=True, padx=25, pady=15)

        # ── RIGHT SIDEBAR ──────────────────────────────────────────────
        right = tk.Frame(cols, bg=BG_DARK, width=320)
        right.pack(side="right", fill="y", padx=(15, 0))
        right.pack_propagate(False)

        # ── BUTTONS always anchored at the bottom first ─────────────────
        btn_frame = tk.Frame(right, bg=BG_DARK)
        btn_frame.pack(side="bottom", fill="x", pady=(6, 4))

        self.suspect_stop_btn = tk.Button(btn_frame, text="■  STOP SEARCH",
                                          font=self.font_btn, bg=ACCENT_RED,
                                          fg=TEXT_PRIMARY, pady=11,
                                          state="disabled",
                                          command=self._suspect_stop)
        self.suspect_stop_btn.pack(fill="x", pady=(0, 6))

        self.suspect_start_btn = tk.Button(btn_frame, text="🔍  START SEARCH",
                                           font=self.font_btn, bg=ACCENT_AMBER,
                                           fg=BG_DARK, pady=11,
                                           command=self._suspect_start)
        self.suspect_start_btn.pack(fill="x")

        # ── Scrollable area for all controls above buttons ──────────────
        ctrl_canvas = tk.Canvas(right, bg=BG_DARK, highlightthickness=0)
        ctrl_scroll = tk.Scrollbar(right, orient="vertical",
                                   command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        # Don't show scrollbar unless needed — pack canvas only
        ctrl_canvas.pack(side="top", fill="both", expand=True)

        ctrl_inner = tk.Frame(ctrl_canvas, bg=BG_DARK)
        ctrl_canvas.create_window((0, 0), window=ctrl_inner, anchor="nw")
        ctrl_inner.bind("<Configure>",
            lambda e: ctrl_canvas.configure(
                scrollregion=ctrl_canvas.bbox("all")))
        # Mouse-wheel scrolling
        def _mw(e):
            ctrl_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        ctrl_canvas.bind_all("<MouseWheel>", _mw)

        # Use ctrl_inner as parent for all cards from here on
        right = ctrl_inner  # rebind local name

        # Source selection
        src_card = tk.Frame(right, bg=BG_CARD, highlightthickness=1,
                            highlightbackground=BORDER)
        src_card.pack(fill="x", pady=(0, 10))
        tk.Label(src_card, text="VIDEO SOURCE", font=self.font_mono,
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        src_inner = tk.Frame(src_card, bg=BG_CARD)
        src_inner.pack(fill="x", padx=10, pady=8)

        tk.Button(src_inner, text="📁  LOAD VIDEO FILE", font=self.font_btn,
                  bg=BG_SURFACE, fg=TEXT_PRIMARY, relief="flat", cursor="hand2",
                  pady=9, command=self._suspect_load_video).pack(fill="x", pady=(0, 6))
        tk.Button(src_inner, text="📷  USE WEBCAM", font=self.font_btn,
                  bg=BG_SURFACE, fg=TEXT_PRIMARY, relief="flat", cursor="hand2",
                  pady=9, command=lambda: self._suspect_set_source(0)).pack(fill="x")

        self.suspect_source_lbl = tk.Label(src_card, text="No source selected",
                                           font=("Courier New", 8), fg=TEXT_DIM,
                                           bg=BG_CARD, wraplength=280)
        self.suspect_source_lbl.pack(pady=(0, 8))

        # Description input
        desc_card = tk.Frame(right, bg=BG_CARD, highlightthickness=1,
                             highlightbackground=BORDER)
        desc_card.pack(fill="x", pady=(0, 10))
        tk.Label(desc_card, text="SUSPECT DESCRIPTION", font=self.font_mono,
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        desc_inner = tk.Frame(desc_card, bg=BG_CARD)
        desc_inner.pack(fill="x", padx=10, pady=8)

        tk.Label(desc_inner, text="Describe clothing, accessories, colours:",
                 font=("Courier New", 8), fg=TEXT_SECONDARY, bg=BG_CARD).pack(anchor="w")

        self.suspect_desc_text = tk.Text(desc_inner, height=4, width=30,
                                         bg=BG_SURFACE, fg=TEXT_PRIMARY,
                                         font=("Courier New", 10),
                                         insertbackground=ACCENT_CYAN,
                                         relief="flat",
                                         highlightthickness=1,
                                         highlightbackground=BORDER_BRIGHT,
                                         wrap="word")
        self.suspect_desc_text.pack(fill="x", pady=(6, 0))
        self.suspect_desc_text.insert("1.0", "e.g. blue jacket, black pants, brown bag")
        self.suspect_desc_text.bind("<FocusIn>", self._suspect_clear_placeholder)

        # Settings
        cfg_card = tk.Frame(right, bg=BG_CARD, highlightthickness=1,
                            highlightbackground=BORDER)
        cfg_card.pack(fill="x", pady=(0, 10))
        tk.Label(cfg_card, text="SEARCH SETTINGS", font=self.font_mono,
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        cfg_inner = tk.Frame(cfg_card, bg=BG_CARD)
        cfg_inner.pack(fill="x", padx=10, pady=8)

        # Threshold slider
        tk.Label(cfg_inner, text="MATCH SENSITIVITY", font=("Courier New", 8, "bold"),
                 fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w")
        thresh_row = tk.Frame(cfg_inner, bg=BG_CARD)
        thresh_row.pack(fill="x", pady=(4, 8))
        self.suspect_thresh_var = tk.DoubleVar(value=0.70)
        thresh_slider = tk.Scale(thresh_row, from_=0.10, to=0.90,
                                 resolution=0.01, orient="horizontal",
                                 variable=self.suspect_thresh_var,
                                 bg=BG_CARD, fg=TEXT_PRIMARY,
                                 troughcolor=BG_SURFACE,
                                 highlightthickness=0,
                                 showvalue=True, length=200)
        thresh_slider.pack(side="left")

        # Frame skip
        tk.Label(cfg_inner, text="FRAME SKIP (1=every frame, 5=every 5th)",
                 font=("Courier New", 8, "bold"), fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w")
        self.suspect_skip_var = tk.IntVar(value=5)
        skip_slider = tk.Scale(cfg_inner, from_=1, to=15,
                               resolution=1, orient="horizontal",
                               variable=self.suspect_skip_var,
                               bg=BG_CARD, fg=TEXT_PRIMARY,
                               troughcolor=BG_SURFACE,
                               highlightthickness=0,
                               showvalue=True, length=200)
        skip_slider.pack(anchor="w")

        # Progress bar
        prog_card = tk.Frame(right, bg=BG_CARD, highlightthickness=1,
                             highlightbackground=BORDER)
        prog_card.pack(fill="x", pady=(0, 10))
        tk.Label(prog_card, text="SEARCH PROGRESS", font=self.font_mono,
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.suspect_prog = ttk.Progressbar(prog_card, orient="horizontal",
                                            mode="determinate")
        self.suspect_prog.pack(fill="x", padx=12, pady=(4, 2))
        self.suspect_prog_lbl = tk.Label(prog_card, text="0%",
                                         font=self.font_mono, fg=TEXT_DIM,
                                         bg=BG_CARD)
        self.suspect_prog_lbl.pack(pady=(0, 6))

        # Match count
        mc_card = tk.Frame(right, bg=BG_CARD, highlightthickness=1,
                           highlightbackground=BORDER)
        mc_card.pack(fill="x", pady=(0, 10))
        tk.Label(mc_card, text="SUSPECTS FOUND", font=self.font_mono,
                 fg=TEXT_SECONDARY, bg=BG_SURFACE).pack(fill="x", pady=6)
        self.suspect_count_lbl = tk.Label(mc_card, text="0",
                                          font=("Courier New", 28, "bold"),
                                          fg=ACCENT_AMBER, bg=BG_CARD)
        self.suspect_count_lbl.pack(pady=6)

        # ── LEFT AREA (Video feed + results grid) ─────────────────────
        left = tk.Frame(cols, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        # Live feed
        feed_wrap = tk.Frame(left, bg=BG_CARD, highlightthickness=1,
                             highlightbackground=BORDER_BRIGHT)
        feed_wrap.pack(fill="both", expand=True)

        feed_topbar = tk.Frame(feed_wrap, bg=BG_SURFACE)
        feed_topbar.pack(fill="x")
        tk.Label(feed_topbar, text="🔍  SUSPECT SEARCH FEED",
                 font=("Courier New", 9, "bold"), fg=ACCENT_AMBER,
                 bg=BG_SURFACE).pack(side="left", padx=12, pady=6)
        self.suspect_frame_lbl = tk.Label(feed_topbar, text="",
                                          font=("Courier New", 8),
                                          fg=TEXT_DIM, bg=BG_SURFACE)
        self.suspect_frame_lbl.pack(side="right", padx=12, pady=6)

        self.suspect_vid_lbl = tk.Label(feed_wrap, bg="#040810",
                                        text="[ LOAD A VIDEO AND START SEARCH ]",
                                        font=self.font_head, fg=TEXT_DIM)
        self.suspect_vid_lbl.pack(fill="both", expand=True, padx=4, pady=4)

        # Results thumbnail strip at bottom
        strip_wrap = tk.Frame(left, bg=BG_PANEL, height=110)
        strip_wrap.pack(fill="x", pady=(8, 0))
        strip_wrap.pack_propagate(False)

        tk.Label(strip_wrap, text="MATCH THUMBNAILS",
                 font=("Courier New", 8, "bold"), fg=TEXT_DIM,
                 bg=BG_PANEL).pack(anchor="w", padx=10, pady=(4, 0))

        strip_scroll_frame = tk.Frame(strip_wrap, bg=BG_PANEL)
        strip_scroll_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.suspect_strip_canvas = tk.Canvas(strip_scroll_frame, bg=BG_PANEL,
                                              height=80, highlightthickness=0)
        strip_hscroll = tk.Scrollbar(strip_scroll_frame, orient="horizontal",
                                     command=self.suspect_strip_canvas.xview)
        self.suspect_strip_canvas.configure(xscrollcommand=strip_hscroll.set)
        self.suspect_strip_canvas.pack(fill="x")
        strip_hscroll.pack(fill="x")

        self.suspect_strip_inner = tk.Frame(self.suspect_strip_canvas, bg=BG_PANEL)
        self.suspect_strip_canvas.create_window((0, 0), window=self.suspect_strip_inner,
                                                anchor="nw")
        self.suspect_strip_inner.bind(
            "<Configure>",
            lambda e: self.suspect_strip_canvas.configure(
                scrollregion=self.suspect_strip_canvas.bbox("all"))
        )

    # ── Suspect Finder helpers ─────────────────────────────────────────
    def _suspect_clear_placeholder(self, event):
        current = self.suspect_desc_text.get("1.0", "end-1c")
        if current.startswith("e.g."):
            self.suspect_desc_text.delete("1.0", "end")

    def _suspect_load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if path:
            self._suspect_set_source(path)

    def _suspect_set_source(self, source):
        self.suspect_video_path = source
        label = "Webcam (live)" if source == 0 else os.path.basename(str(source))
        self.suspect_source_lbl.config(text=f"✓ {label}", fg=ACCENT_GREEN)

    def _suspect_start(self):
        if not hasattr(self, 'suspect_video_path') or self.suspect_video_path is None:
            messagebox.showwarning("No Source", "Please load a video file or select webcam first.")
            return

        desc = self.suspect_desc_text.get("1.0", "end-1c").strip()
        if not desc or desc.startswith("e.g."):
            messagebox.showwarning("No Description", "Please enter a suspect description.")
            return

        # Reset state
        self.suspect_is_running = True
        self.suspect_match_results = []
        self.suspect_count_lbl.config(text="0")
        self.suspect_prog["value"] = 0
        self.suspect_prog_lbl.config(text="0%", fg=ACCENT_CYAN)
        self.suspect_phase_lbl.config(text="[ SEARCHING... ]", fg=ACCENT_AMBER)
        self._set_sidebar_status("searching", "Suspect search active")

        # Clear thumbnail strip
        for w in self.suspect_strip_inner.winfo_children():
            w.destroy()

        self.suspect_start_btn.config(state="disabled", bg=TEXT_DIM)
        self.suspect_stop_btn.config(state="normal")

        threshold = self.suspect_thresh_var.get()
        skip = self.suspect_skip_var.get()

        threading.Thread(
            target=self._suspect_search_loop,
            args=(self.suspect_video_path, desc, threshold, skip),
            daemon=True
        ).start()

    def _suspect_stop(self):
        self.suspect_is_running = False
        self.suspect_start_btn.config(state="normal", bg=ACCENT_AMBER)
        self.suspect_stop_btn.config(state="disabled")
        self.suspect_phase_lbl.config(text="[ STOPPED ]", fg=ACCENT_RED)
        self._set_sidebar_status("idle", "Suspect search stopped")

    def _suspect_search_loop(self, source, description, threshold, skip_frames):
        """Background thread: streams video, runs YOLO + colour matching per frame."""
        try:
            from suspect_finder import parse_description, YOLODetector, find_matches
        except ImportError:
            self.after(0, lambda: messagebox.showerror(
                "Missing Module",
                "suspect_finder.py not found in the same directory as main.py."))
            self.after(0, self._suspect_stop)
            return

        # Parse description once (may call Groq API)
        self.after(0, lambda: self.suspect_phase_lbl.config(
            text="[ PARSING DESCRIPTION... ]", fg=ACCENT_CYAN))

        # Pull groq key from env or leave blank for rule-based
        groq_key = GROQ_API_KEY
        if groq_key:
            print("[Suspect Finder] Using Groq LLaMA API for description parsing.")
        else:
            print("[Suspect Finder] No GROQ_API_KEY found — using rule-based parser.")
        try:
            attributes = parse_description(description, groq_key)
        except Exception as e:
            attributes = {}

        detector = YOLODetector(model_size="yolov8n.pt", confidence=0.40)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror("Error", "Cannot open video source."))
            self.after(0, self._suspect_stop)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = 0
        match_count = 0

        self.after(0, lambda: self.suspect_phase_lbl.config(
            text="[ SCANNING FOOTAGE ]", fg=ACCENT_AMBER))

        os.makedirs(SUSPECT_FINDER_DIR, exist_ok=True)

        while cap.isOpened() and self.suspect_is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Update progress
            pct = min(int((frame_count / total_frames) * 99), 99)
            if frame_count % 10 == 0:
                self.after(0, lambda p=pct: self._suspect_update_progress(p))

            # Only process every Nth frame
            if frame_count % skip_frames != 0:
                continue

            timestamp = frame_count / fps

            # Detect persons in this frame
            detections = detector.detect_frame(frame, frame_count, timestamp)
            if not detections:
                continue

            # Score against description
            results = find_matches(detections, attributes,
                                   threshold=threshold, top_n=50)

            # Show annotated frame on screen
            annotated = self._suspect_annotate(frame.copy(), detections, results)
            ts_str = f"Frame {frame_count}  |  {int(timestamp//60):02d}:{int(timestamp%60):02d}"
            self.after(0, lambda f=annotated, t=ts_str: self._suspect_display(f, t))

            # Save new matches and add thumbnails
            for r in results:
                match_count += 1
                crop = r.crop
                fname = f"suspect_{match_count:04d}_f{r.frame_num}_s{r.overall_score:.2f}.jpg"
                fpath = os.path.join(SUSPECT_FINDER_DIR, fname)
                import cv2 as _cv2
                _cv2.imwrite(fpath, crop)
                self.suspect_match_results.append(r)
                self.after(0, lambda c=crop.copy(), sc=r.overall_score:
                           self._suspect_add_thumbnail(c, sc))
                self.after(0, lambda n=match_count:
                           self.suspect_count_lbl.config(text=str(n)))

        cap.release()
        # Final update
        self.after(0, lambda: self._suspect_update_progress(100))
        self.after(0, lambda: self.suspect_phase_lbl.config(
            text=f"[ COMPLETE — {match_count} MATCH(ES) ]", fg=ACCENT_GREEN))
        self.after(0, lambda: self._set_sidebar_status(
            "done", f"Suspect search done — {match_count} found"))
        # Refresh suspect tab count in Results page
        if hasattr(self, '_tab_count_labels') and "suspect" in self._tab_count_labels:
            self.after(0, lambda n=match_count: self._tab_count_labels["suspect"].config(text=str(n)))
        self.after(0, lambda: self.suspect_start_btn.config(
            state="normal", bg=ACCENT_AMBER))
        self.after(0, lambda: self.suspect_stop_btn.config(state="disabled"))
        self.suspect_is_running = False

    def _suspect_update_progress(self, value):
        self.suspect_prog["value"] = value
        color = ACCENT_CYAN if value < 100 else ACCENT_GREEN
        self.suspect_prog_lbl.config(text=f"{value}%", fg=color)

    def _suspect_annotate(self, frame, detections, matches):
        """Draw bounding boxes on frame: amber=match, dim=no match."""
        match_frames = {r.frame_num for r in matches}
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_match = any(r.frame_num == det.frame_num and r.bbox == det.bbox
                           for r in matches)
            if is_match:
                match_r = next(r for r in matches
                               if r.frame_num == det.frame_num and r.bbox == det.bbox)
                score = match_r.overall_score
                color = (0, 200, 255) if score >= 0.35 else (0, 140, 255)
                # Build label with skin tone if available
                st = getattr(match_r, 'skin_tone_info', {})
                if st and st.get('tone') and st['tone'] != 'unknown' and st.get('confidence', 0) >= 0.25:
                    label = f"SUSPECT {score:.0%} | {st['tone']} ({st['confidence']:.0%})"
                else:
                    label = f"SUSPECT {score:.0%}"
                thick = 2
            else:
                color = (60, 60, 60)
                label = ""
                thick = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            if label:
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame

    def _suspect_display(self, frame, ts_str):
        self._apply_image_to_label(frame, self.suspect_vid_lbl)
        self.suspect_frame_lbl.config(text=ts_str)

    def _suspect_add_thumbnail(self, crop, score):
        """Add a match thumbnail to the horizontal strip."""
        try:
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img.thumbnail((72, 80))
            bg = Image.new("RGB", (72, 80), (20, 20, 30))
            ox = (72 - img.width) // 2
            oy = (80 - img.height) // 2
            bg.paste(img, (ox, oy))
            img_tk = ImageTk.PhotoImage(bg)

            frame = tk.Frame(self.suspect_strip_inner, bg=BG_PANEL,
                             highlightthickness=1,
                             highlightbackground=ACCENT_AMBER if score >= 0.35 else BORDER)
            frame.pack(side="left", padx=3, pady=2)

            lbl = tk.Label(frame, image=img_tk, bg=BG_PANEL)
            lbl.image = img_tk
            lbl.pack()

            score_lbl = tk.Label(frame, text=f"{score:.0%}",
                                 font=("Courier New", 7, "bold"),
                                 fg=ACCENT_AMBER if score >= 0.35 else TEXT_SECONDARY,
                                 bg=BG_PANEL)
            score_lbl.pack()
        except Exception:
            pass


if __name__ == "__main__":
    app = SentinelVision()
    app.mainloop()