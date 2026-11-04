"""View one raw 2-D ion image; only the display contrast is adjusted.

Run: python view_ion.py
     python view_ion.py 20260404_2000pure --index 10
     python view_ion.py --save preview.png --no-show
Requires numpy and matplotlib. No detection, filtering, or data rewriting.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib


def find_images(source):
    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".npy":
            raise ValueError("Input file must be a .npy image.")
        return [source]
    if not source.is_dir():
        raise ValueError("Input does not exist: {}".format(source))
    files = sorted(source.rglob("*.npy"))
    if not files:
        raise ValueError("No .npy images found in {}".format(source))
    return files


def load_image(path):
    data = np.load(str(path), allow_pickle=False)
    if data.ndim != 2 or not data.size or data.dtype.kind not in "buif":
        raise ValueError("Expected a nonempty, real 2-D array, got {} / {}: {}"
                         .format(data.shape, data.dtype, path))
    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN or infinity: {}".format(path))
    return data


def contrast_limits(data, percentiles):
    low, high = (float(v) for v in np.percentile(data, percentiles))
    if low == high:
        low, high = float(data.min()), float(data.max())
    if low == high:
        high = low + max(1.0, abs(low) * 1e-6)
    return low, high


class IonViewer:
    def __init__(self, files, args):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RangeSlider

        self.plt = plt
        self.files = files
        self.args = args
        self.index = args.index
        self.data = load_image(files[self.index])
        self.fig, self.ax = plt.subplots(figsize=(14, 5.5))
        self.fig.subplots_adjust(left=0.065, right=0.89, bottom=0.32, top=0.82)
        self.fig.canvas.manager.set_window_title("CCD ion image viewer")
        low, high = self.initial_limits()
        self.artist = self.ax.imshow(
            self.data, cmap=args.cmap, origin="upper", interpolation="nearest",
            aspect="equal", vmin=low, vmax=high,
        )
        self.ax.set_xlabel("x / column (pixel)")
        self.ax.set_ylabel("y / row (pixel)")
        self.ax.format_coord = self.pixel_value
        cax = self.fig.add_axes([0.915, 0.36, 0.015, 0.40])
        self.colorbar = self.fig.colorbar(self.artist, cax=cax)
        self.colorbar.set_label("Raw intensity (stored units)")
        self.heading = self.fig.text(0.065, 0.94, "", fontsize=12, weight="bold")
        self.stats = self.fig.text(0.065, 0.89, "", fontsize=10)
        self.message = self.fig.text(0.065, 0.025, "", fontsize=9)
        self.fig.text(0.065, 0.075,
                      "Scroll: zoom at cursor | Toolbar: pan / rectangle zoom | "
                      "Left/Right: previous/next | R: reset | S: save", fontsize=9)
        self.slider_ax = self.fig.add_axes([0.15, 0.21, 0.70, 0.045])
        self.RangeSlider = RangeSlider
        self.slider = None
        self.setup_slider(low, high)
        labels = ["Previous", "Next", "Reset view", "Auto contrast", "Full range", "Gray / color", "Save PNG"]
        actions = [lambda _: self.change_image(-1), lambda _: self.change_image(1),
                   self.reset_view, self.auto_contrast, self.full_range,
                   self.toggle_color, lambda _: self.save()]
        self.buttons = []
        for i, (label, action) in enumerate(zip(labels, actions)):
            button_ax = self.fig.add_axes([0.065 + i * 0.128, 0.12, 0.118, 0.055])
            button = Button(button_ax, label)
            button.on_clicked(action)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("scroll_event", self.zoom)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress)
        self.update_text()

    def initial_limits(self):
        low, high = contrast_limits(self.data, self.args.percentiles)
        low = self.args.vmin if self.args.vmin is not None else low
        high = self.args.vmax if self.args.vmax is not None else high
        if high <= low:
            raise ValueError("Display vmax must exceed vmin (including automatic limits).")
        return low, high

    def setup_slider(self, low, high):
        if self.slider is not None:
            self.slider.disconnect_events()
        self.slider_ax.clear()
        lower = min(float(self.data.min()), low)
        upper = max(float(self.data.max()), high)
        self.slider = self.RangeSlider(self.slider_ax, "Display range", lower, upper,
                                       valinit=(low, high), valfmt="%.1f")
        self.slider.on_changed(self.set_contrast)

    def set_contrast(self, values):
        low, high = values
        if high <= low:
            return
        self.artist.set_clim(low, high)
        self.fig.canvas.draw_idle()

    def update_text(self):
        path = self.files[self.index]
        height, width = self.data.shape
        self.heading.set_text("[{} / {}] {} / {}".format(
            self.index + 1, len(self.files), path.parent.name, path.name))
        self.stats.set_text("{} x {} pixels (W x H) | {} | raw min {:.3f}, max {:.3f} | "
                            "Display contrast only; original pixels unchanged".format(
                                width, height, self.data.dtype,
                                float(self.data.min()), float(self.data.max())))
        print("{}\n  shape={} dtype={} raw=[{:.6g}, {:.6g}]".format(
            path, self.data.shape, self.data.dtype,
            float(self.data.min()), float(self.data.max())), flush=True)

    def pixel_value(self, x, y):
        col, row = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
            return "x={}  y={}  raw={:.6g}".format(col, row, self.data[row, col])
        return "x={:.1f}  y={:.1f}".format(x, y)

    def reset_view(self, _=None):
        height, width = self.data.shape
        self.ax.set_xlim(-0.5, width - 0.5)
        self.ax.set_ylim(height - 0.5, -0.5)
        self.fig.canvas.draw_idle()

    def auto_contrast(self, _=None):
        self.slider.set_val(contrast_limits(self.data, self.args.percentiles))

    def full_range(self, _=None):
        self.slider.set_val(contrast_limits(self.data, (0, 100)))

    def toggle_color(self, _=None):
        self.artist.set_cmap("inferno" if self.artist.get_cmap().name == "gray" else "gray")
        self.fig.canvas.draw_idle()

    def change_image(self, step):
        index = self.index + step
        if not 0 <= index < len(self.files):
            self.message.set_text("Already at the first / last image.")
            self.fig.canvas.draw_idle()
            return
        try:
            data = load_image(self.files[index])
        except (ValueError, OSError) as exc:
            self.message.set_text(str(exc))
            self.fig.canvas.draw_idle()
            return
        old_data = self.data
        self.data = data
        try:
            low, high = self.initial_limits()
        except ValueError as exc:
            self.data = old_data
            self.message.set_text(str(exc))
            self.fig.canvas.draw_idle()
            return
        self.index = index
        self.artist.set_data(data)
        height, width = data.shape
        self.artist.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
        self.artist.set_clim(low, high)
        self.setup_slider(low, high)
        self.update_text()
        self.message.set_text("")
        self.reset_view()

    def zoom(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        for get_limits, set_limits, center in (
            (self.ax.get_xlim, self.ax.set_xlim, event.xdata),
            (self.ax.get_ylim, self.ax.set_ylim, event.ydata),
        ):
            start, end = get_limits()
            set_limits(center + (start - center) * factor, center + (end - center) * factor)
        self.fig.canvas.draw_idle()

    def keypress(self, event):
        # Arrow keys otherwise trigger Matplotlib's navigation-history shortcuts.
        if event.key == "right":
            self.change_image(1)
        elif event.key == "left":
            self.change_image(-1)
        elif event.key == "r":
            self.reset_view()
        elif event.key == "s":
            self.save()

    def save(self, destination=None):
        source = self.files[self.index]
        if destination is None:
            destination = Path(__file__).resolve().parent / "previews" / (source.stem + "_preview.png")
        destination = Path(destination).expanduser().resolve()
        if destination.suffix.lower() != ".png":
            raise ValueError("Preview output must end in .png.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Export the current view without interactive widgets; retain raw-unit colorbar.
        fig, ax = self.plt.subplots(figsize=(14, 4), constrained_layout=True)
        artist = ax.imshow(self.data, origin="upper", interpolation="nearest", aspect="equal",
                           cmap=self.artist.get_cmap(), vmin=self.artist.norm.vmin,
                           vmax=self.artist.norm.vmax)
        ax.set_xlim(self.ax.get_xlim())
        ax.set_ylim(self.ax.get_ylim())
        ax.set(xlabel="x / column (pixel)", ylabel="y / row (pixel)",
               title="{} / {}".format(source.parent.name, source.name))
        fig.colorbar(artist, ax=ax, shrink=0.8, label="Raw intensity (stored units)")
        fig.savefig(str(destination), dpi=180)
        self.plt.close(fig)
        self.message.set_text("Saved: {}".format(destination))
        self.fig.canvas.draw_idle()
        print("Saved: {}".format(destination), flush=True)
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                        help=".npy image or directory (searched recursively); default: script directory")
    parser.add_argument("--index", type=int, default=0, help="0-based image index in sorted file list")
    parser.add_argument("--cmap", default="gray", choices=("gray", "inferno", "viridis", "magma"))
    parser.add_argument("--percentiles", type=float, nargs=2, default=(1, 99.9), metavar=("LOW", "HIGH"),
                        help="display percentile limits, default: 1 99.9; use 0 100 for full range")
    parser.add_argument("--vmin", type=float, help="fixed lower display limit, in stored intensity units")
    parser.add_argument("--vmax", type=float, help="fixed upper display limit, in stored intensity units")
    parser.add_argument("--save", type=Path, help="save a PNG preview at startup")
    parser.add_argument("--no-show", action="store_true", help="export without opening a window; requires --save")
    args = parser.parse_args()
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100:
        parser.error("percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if any(v is not None and not np.isfinite(v) for v in (args.vmin, args.vmax)):
        parser.error("vmin/vmax must be finite")
    if args.no_show and not args.save:
        parser.error("--no-show requires --save output.png")
    if args.no_show:
        matplotlib.use("Agg")
    for key in ("keymap.back", "keymap.forward", "keymap.save", "keymap.home"):
        matplotlib.rcParams[key] = []
    try:
        files = find_images(args.source)
        if not 0 <= args.index < len(files):
            raise ValueError("--index must be between 0 and {}".format(len(files) - 1))
        viewer = IonViewer(files, args)
        if args.save:
            viewer.save(args.save)
        if not args.no_show:
            viewer.plt.show()
        else:
            viewer.plt.close(viewer.fig)
    except (ValueError, OSError) as exc:
        parser.exit(1, "Error: {}\n".format(exc))


if __name__ == "__main__":
    main()
