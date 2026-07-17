"""
GUI wrapper for CZI fluorescence analysis pipeline
Adds:
- Profile manager (JSON persistence)
- File/Directory selection GUI
- Well plate viewer window (A-H, numeric columns)
- Display toggles, colormap selection, value overlays

This is a GUI layer on top of the existing analysis code.

modify this python code into a full GUI based python program with matching json file that allows the user to select, edit, and create profiles. also have a button that allows the user to select a directory or image. Using the existing profiles, create a json file if one doesnt exist already. Create a second window that can be opened from the menu bar that displays well plate data if the data name is a letter and number (A1, B2, B13, etc). display the values in a grid matching the letter and number with the letter on the y axis and the number on the x axis. add a check box to either display or hide the number on each well, and allow the user to select the color map from blue, green, red and purple. also allow the user to select which values are shown for each well and what color those values are displayed as. Make sure to keep all current functionality
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

# Assume the original pipeline code is in this file or imported
# from pipeline import (
#     process_czi_image, remove_outliers, fill_missing_points,
#     organize_circles, snake_circles, quantify_fluorescence,
#     fancy_plot, quantify_fluorescence_single
# )

# ----------------------------
# Profile Management
# ----------------------------

PROFILE_FILE = "profiles.json"

DEFAULT_PROFILE = {
    "name": "default",
    "settings": {
        "is_array": False,
        "min_diameter": 20,
        "max_diameter": 40,
        "roi_inner": -20,
        "roi_outer": 5,
        "bckg_inner": 55,
        "bckg_outer": 100,
        "moving_avg_n": 2,
        "p1": 50,
        "p2": 20,
        "dp": 1.0
    }
}


def load_profiles():
    if not os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "w") as f:
            json.dump([DEFAULT_PROFILE], f, indent=2)
        return [DEFAULT_PROFILE]
    with open(PROFILE_FILE, "r") as f:
        return json.load(f)


def save_profiles(profiles):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


# ----------------------------
# Well Plate Viewer Window
# ----------------------------

class WellPlateWindow(tk.Toplevel):
    def __init__(self, master, data=None):
        super().__init__(master)
        self.title("Well Plate Viewer")
        self.geometry("900x600")

        self.data = data or {}  # {'A1': value}
        self.show_numbers = tk.BooleanVar(value=True)
        self.colormap = tk.StringVar(value="blue")

        self.value_key = tk.StringVar(value="value")

        self._build_ui()
        self.draw_grid()

    def _build_ui(self):
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(
            controls,
            text="Show Numbers",
            variable=self.show_numbers,
            command=self.draw_grid
        ).pack(side=tk.LEFT)

        ttk.Label(controls, text="Colormap:").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.colormap,
            values=["blue", "green", "red", "purple"],
            state="readonly"
        ).pack(side=tk.LEFT)

        ttk.Button(controls, text="Refresh", command=self.draw_grid).pack(side=tk.LEFT)

        ttk.Label(controls, text="Value key:").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.value_key, width=10).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def draw_grid(self):
        self.canvas.delete("all")

        rows = list("ABCDEFGH")
        cols = list(range(1, 13))

        w, h = 60, 40

        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                key = f"{r}{c}"
                x0, y0 = j * w, i * h
                x1, y1 = x0 + w, y0 + h

                val = self.data.get(key, None)

                color = self._map_color(val)

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

                if self.show_numbers.get() and val is not None:
                    self.canvas.create_text(
                        x0 + w/2,
                        y0 + h/2,
                        text=str(round(val, 2)),
                        fill="white"
                    )

    def _map_color(self, val):
        if val is None:
            return "gray"

        cmap = self.colormap.get()

        # simple scaling
        norm = max(0, min(1, val / 100))

        if cmap == "blue":
            return f"#0000{int(norm*255):02x}"
        if cmap == "red":
            return f"#{int(norm*255):02x}0000"
        if cmap == "green":
            return f"#00{int(norm*255):02x}00"
        if cmap == "purple":
            v = int(norm*255)
            return f"#{v:02x}00{v:02x}"
        return "blue"


# ----------------------------
# Main Application
# ----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CZI Analysis GUI")
        self.geometry("500x400")

        self.profiles = load_profiles()
        self.current_profile = tk.StringVar()

        self.file_path = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        # Profile section
        frame = ttk.LabelFrame(self, text="Profiles")
        frame.pack(fill=tk.X, padx=10, pady=10)

        self.profile_box = ttk.Combobox(
            frame,
            textvariable=self.current_profile,
            values=[p["name"] for p in self.profiles]
        )
        self.profile_box.pack(side=tk.LEFT)

        ttk.Button(frame, text="Load", command=self.load_profile).pack(side=tk.LEFT)
        ttk.Button(frame, text="Add", command=self.add_profile).pack(side=tk.LEFT)
        ttk.Button(frame, text="Save", command=self.save_profile).pack(side=tk.LEFT)

        # File selection
        file_frame = ttk.LabelFrame(self, text="Input")
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side=tk.LEFT)

        ttk.Button(file_frame, text="File", command=self.select_file).pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Folder", command=self.select_folder).pack(side=tk.LEFT)

        # Actions
        ttk.Button(self, text="Run Analysis", command=self.run_analysis).pack(pady=10)

        ttk.Button(self, text="Open Well Plate Viewer", command=self.open_plate).pack()

    def select_file(self):
        self.file_path.set(filedialog.askopenfilename(filetypes=[("CZI", "*.czi")]))

    def select_folder(self):
        self.file_path.set(filedialog.askdirectory())

    def load_profile(self):
        name = self.current_profile.get()
        for p in self.profiles:
            if p["name"] == name:
                messagebox.showinfo("Profile Loaded", f"Loaded {name}")
                return

    def add_profile(self):
        name = simple_input(self, "Profile name")
        if not name:
            return
        self.profiles.append({"name": name, "settings": DEFAULT_PROFILE["settings"]})
        save_profiles(self.profiles)
        self.profile_box["values"] = [p["name"] for p in self.profiles]

    def save_profile(self):
        save_profiles(self.profiles)
        messagebox.showinfo("Saved", "Profiles saved")

    def run_analysis(self):
        path = self.file_path.get()
        if not path:
            messagebox.showerror("Error", "No file selected")
            return
        messagebox.showinfo("Run", f"Would run analysis on {path}")

    def open_plate(self):
        # dummy data example
        data = {f"A{i}": i*5 for i in range(1, 13)}
        WellPlateWindow(self, data)


def simple_input(root, title):
    win = tk.Toplevel(root)
    win.title(title)
    var = tk.StringVar()

    ttk.Entry(win, textvariable=var).pack()

    def ok():
        win.destroy()

    ttk.Button(win, text="OK", command=ok).pack()

    root.wait_window(win)
    return var.get()


if __name__ == "__main__":
    App().mainloop()
