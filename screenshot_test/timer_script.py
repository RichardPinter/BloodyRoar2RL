#!/usr/bin/env python3
"""
region_overlay.py

Creates a transparent, click-through overlay window covering a specific screen region
and displays a live‐updating timestamp in its top-right corner.

Dependencies:
  - Python 3.7+
  - tkinter (standard library)
  - ctypes (standard library)

Usage:
  python region_overlay.py
Press ESC in this console or close the window to exit.
"""

import time
import ctypes
import tkinter as tk
import sys

# ─── CONFIG ────────────────────────────────────────────────────────────────
# REGION = (x, y, width, height)
REGION = (0, 0, 624, 548)

# ─── Win32 API for click-through and transparency ──────────────────────────
GWL_EXSTYLE        = -20
WS_EX_LAYERED      = 0x00080000
WS_EX_TRANSPARENT  = 0x00000020
LWA_COLORKEY       = 0x00000001

user32 = ctypes.windll.user32
SetWindowLong    = user32.SetWindowLongW
GetWindowLong    = user32.GetWindowLongW
SetLayeredWindowAttributes = user32.SetLayeredWindowAttributes

def make_clickthrough(hwnd):
    """
    Configure the window to be layered+transparent to mouse events,
    and treat white (#FFFFFF) as fully transparent.
    """
    ex = GetWindowLong(hwnd, GWL_EXSTYLE)
    ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    SetWindowLong(hwnd, GWL_EXSTYLE, ex)
    # White (0xFFFFFF) becomes the transparent colorkey
    SetLayeredWindowAttributes(hwnd, 0xFFFFFF, 0, LWA_COLORKEY)

# ─── Overlay Application ───────────────────────────────────────────────────
class RegionOverlay(tk.Tk):
    def __init__(self, region):
        super().__init__()
        x0, y0, w, h = region

        # Remove border/title bar and keep on top
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        # Set window geometry to the region
        self.geometry(f"{w}x{h}+{x0}+{y0}")

        # White background will be transparent
        self.config(bg="#FFFFFF")
        self.update_idletasks()

        # Make the window click-through at the Win32 level
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        make_clickthrough(hwnd)

        # Canvas to draw timestamp
        self.canvas = tk.Canvas(self, width=w, height=h, highlightthickness=0, bg="#FFFFFF")
        self.canvas.pack()

        # Create a text item; we'll update its position and content each frame
        self.text_id = self.canvas.create_text(
            w - 5, 5,
            text="",
            anchor="ne",
            font=("Consolas", 18),
            fill="lime"
        )

        # Start the update loop
        self.after(0, self.update_timestamp)

    def update_timestamp(self):
        # Use high-resolution counter for relative timestamp
        ts = time.perf_counter()
        label = f"{ts:0.3f}s"

        # Move text to top-right corner (5px padding)
        w = REGION[2]
        self.canvas.coords(self.text_id, w - 5, 5)
        self.canvas.itemconfig(self.text_id, text=label)

        # Schedule next update ~60 FPS
        self.after(16, self.update_timestamp)

    def run(self):
        try:
            self.mainloop()
        except KeyboardInterrupt:
            pass

# ─── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    overlay = RegionOverlay(REGION)
    overlay.run()
    print("Overlay closed. Exiting.")
    sys.exit(0)
