# plot_utils.py
"""
Centralized plotting utilities (reusable across  pipeline).

Design goals:
- Pure plotting: functions/classes operate on arrays / already-loaded data.
- Minimal assumptions about filesystem or project structure.
- Reuse plot patterns across datared / azimint / calibration / fitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import inspect

from . import general_utils  
from .paths import AnalysisPaths


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# ============================================================
# Style / config
# ============================================================

def figures_dir(
    base_dir: Union[str, Path],
    *,
    figures_subdir: str = "figures",
) -> Path:
    """
    Return a standard figures folder inside a base directory.
    Example: base_dir=<analysis_dir> -> <analysis_dir>/figures
    """
    return Path(base_dir) / str(figures_subdir)


def build_save_kwargs(
    *,
    save: bool = False,
    base_dir: Optional[Union[str, Path]] = None,
    figures_subdir: str = "figures",
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    overwrite: bool = True,
) -> Dict[str, object]:
    """
    Generic passthrough kwargs for plotters.

    Rules:
      - If save=False -> returns kwargs that do nothing.
      - If save=True  -> requires base_dir, saves to <base_dir>/<figures_subdir>.
      - overwrite defaults to True.
    """
    if not save:
        return dict(
            save=False,
            save_dir=None,
            save_name=None,
            save_format=str(save_format),
            save_dpi=int(save_dpi),
            save_overwrite=False,
        )

    if base_dir is None:
        raise ValueError("build_save_kwargs(save=True) requires base_dir=...")

    out_dir = figures_dir(base_dir, figures_subdir=str(figures_subdir))
    return dict(
        save=True,
        save_dir=out_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(overwrite),
    )


def _sanitize_stem(name: str) -> str:
    name = str(name or "").strip()
    if not name:
        return "figure"
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "", name)
    name = name.strip("._-")
    return name or "figure"


def _next_available_path(path: Path, overwrite: bool) -> Path:
    if overwrite or (not path.exists()):
        return path
    stem, suf = path.stem, path.suffix
    for k in range(1, 10000):
        cand = path.with_name(f"{stem}_{k:02d}{suf}")
        if not cand.exists():
            return cand
    return path.with_name(f"{stem}_XXXX{suf}")


def save_figure(
    fig,
    *,
    save_dir: Union[str, Path],
    save_name: str = "figure",
    fmt: str = "png",
    dpi: int = 400,
    overwrite: bool = False,
) -> Path:
    """
    Save helper.
    - If overwrite=False and file exists -> auto-increment suffix.
    - Returns the final Path.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = str(fmt).lstrip(".").lower() or "png"
    stem = _sanitize_stem(save_name)
    out = out_dir / f"{stem}.{fmt}"
    out = _next_available_path(out, bool(overwrite))

    fig.savefig(str(out), dpi=int(dpi), format=fmt, bbox_inches="tight")
    return out


@dataclass(frozen=True)
class PlotStyle:
    title_fontsize: int = 15
    overall_fontsize: int = 13
    label_fontsize: int = 14
    marker_size: float = 5.0  # scatter area (points^2) in most of this module

    def apply(self) -> None:
        plt.rcParams.update({"font.size": self.overall_fontsize})


DEFAULT_STYLE = PlotStyle()
DEFAULT_STYLE.apply()

# ============================================================
# Small generic helpers
# ============================================================

def _diff_on_ref_grid(
    q_ref: np.ndarray,
    I_ref: np.ndarray,
    q: np.ndarray,
    I: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate I(q) onto the reference q-grid over the overlap region and compute dI.
    """
    q_ref = np.asarray(q_ref)
    I_ref = np.asarray(I_ref)
    q = np.asarray(q)
    I = np.asarray(I)

    qmin = max(float(np.min(q_ref)), float(np.min(q)))
    qmax = min(float(np.max(q_ref)), float(np.max(q)))
    if qmax <= qmin:
        return q_ref * np.nan, q_ref * np.nan

    mref = (q_ref >= qmin) & (q_ref <= qmax)
    qg = q_ref[mref]
    I_on_ref = np.interp(qg, q, I)
    dI = I_on_ref - I_ref[mref]
    return qg, dI


# ============================================================
# Clickable legend toggling
# ============================================================

def _make_legend_clickable(
    ax: plt.Axes,
    *,
    legend: Optional[plt.Legend] = None,
    label_to_artists: Optional[Dict[str, List[object]]] = None,
    alpha_inactive: float = 0.25,
    pickradius: float = 5.0,
) -> None:
    """
    Make legend entries clickable: clicking toggles visibility of the corresponding artist(s).
    """
    if legend is None:
        legend = ax.get_legend()
    if legend is None:
        return

    fig = ax.figure

    # Default mapping: label -> [handle]
    if label_to_artists is None:
        handles, labels = ax.get_legend_handles_labels()
        tmp: Dict[str, List[object]] = {}
        for h, lab in zip(handles, labels):
            lab = str(lab)
            tmp.setdefault(lab, []).append(h)
        label_to_artists = tmp

    # Legend handle attribute differs across mpl versions
    leg_handles = getattr(legend, "legend_handles", None)
    if leg_handles is None:
        leg_handles = getattr(legend, "legendHandles", None)
    if leg_handles is None:
        leg_handles = []

    leg_texts = legend.get_texts()

    pick_map: Dict[object, List[object]] = {}

    for i in range(min(len(leg_texts), len(leg_handles))):
        txt = leg_texts[i]
        h = leg_handles[i]
        label = str(txt.get_text())

        targets = label_to_artists.get(label, None)
        if not targets:
            continue
        targets_list = list(targets)

        # initial alpha matches visibility
        vis = True
        for t in targets_list:
            if hasattr(t, "get_visible"):
                vis = bool(t.get_visible())
                break
        for obj in (h, txt):
            try:
                obj.set_alpha(1.0 if vis else float(alpha_inactive))
            except Exception:
                pass

        # enable picking
        for obj in (h, txt):
            try:
                obj.set_picker(True)
                obj.set_pickradius(float(pickradius))
            except Exception:
                try:
                    obj.set_picker(float(pickradius))
                except Exception:
                    pass

        pick_map[h] = targets_list
        pick_map[txt] = targets_list

    if not pick_map:
        return

    def _on_pick(event):
        artist = event.artist
        if artist not in pick_map:
            return

        targets = pick_map[artist]

        base_vis = True
        for t in targets:
            if hasattr(t, "get_visible"):
                base_vis = bool(t.get_visible())
                break
        new_vis = not base_vis

        for t in targets:
            if hasattr(t, "set_visible"):
                t.set_visible(new_vis)

        # update corresponding legend entry alpha
        try:
            if hasattr(artist, "get_text"):
                label = str(artist.get_text())
            else:
                label = None
                for j in range(min(len(leg_texts), len(leg_handles))):
                    if leg_handles[j] is artist:
                        label = str(leg_texts[j].get_text())
                        break

            if label is not None:
                for j in range(min(len(leg_texts), len(leg_handles))):
                    if str(leg_texts[j].get_text()) == label:
                        try:
                            leg_handles[j].set_alpha(1.0 if new_vis else float(alpha_inactive))
                        except Exception:
                            pass
                        try:
                            leg_texts[j].set_alpha(1.0 if new_vis else float(alpha_inactive))
                        except Exception:
                            pass
                        break
        except Exception:
            pass

        fig.canvas.draw_idle()

    cids = getattr(fig, "_legend_toggle_cids", None)
    if cids is None:
        fig._legend_toggle_cids = []
        cids = fig._legend_toggle_cids

    cid = fig.canvas.mpl_connect("pick_event", _on_pick)
    cids.append(cid)


def _markersize_from_scatter_s(s: float) -> float:
    """
    Convert scatter 's' (points^2) to Line2D markersize (points).
    """
    s = float(s)
    return float(np.sqrt(max(s, 1.0)))


def _flatten_errorbar_container(eb) -> List[object]:
    """
    Flatten a Matplotlib ErrorbarContainer into artists (markers + bars + caps).
    """
    artists: List[object] = []
    if eb is None:
        return artists
    lines = getattr(eb, "lines", None)
    if lines is None:
        return artists

    for item in lines:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            for sub in item:
                if sub is not None:
                    artists.append(sub)
        else:
            artists.append(item)
    return artists


# ============================================================
# 2D image plotting
# ============================================================

class Image2DPlotter:
    """
    Quick display of 2D detector images.
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        self.style = style

    def plot(
        self,
        img: np.ndarray,
        *,
        clim: Optional[Tuple[float, float]] = (0.0, 0.05),
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
        show_colorbar: bool = True,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()
        img = np.asarray(img)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if clim is None:
            im = ax.imshow(img)
        else:
            im = ax.imshow(img, clim=clim)
        if show_colorbar:
            fig.colorbar(im, ax=ax)

        if title is not None:
            ax.set_title(title, fontsize=self.style.title_fontsize)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        plt.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("Image2DPlotter.plot(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "image2d"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax

# ============================================================
# Delay distribution plots
# ============================================================

class DelayDistributionPlotter:
    """
    Input:
        delays_by_scan = {scan_number: np.ndarray([...delay values...]), ...}
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        self.style = style

    @staticmethod
    def _validate_inputs(
        delays_by_scan: Dict[int, np.ndarray],
        *,
        mode: str,
        view: str,
    ) -> None:
        if mode not in ("overlay", "per_scan"):
            raise ValueError("mode must be 'overlay' or 'per_scan'")
        if view not in ("scatter", "hist"):
            raise ValueError("view must be 'scatter' or 'hist'")
        if not isinstance(delays_by_scan, dict) or len(delays_by_scan) == 0:
            raise ValueError("delays_by_scan must be a non-empty dict {scan: delays_array}")

    def plot(
        self,
        delays_by_scan: Dict[int, np.ndarray],
        *,
        mode: str = "overlay",
        view: str = "scatter",
        unit: str = "ps",
        bins: int = 200,
        show_median: bool = True,
        alpha: float = 0.6,
        ms: float = 3.0,
        title: Optional[str] = None,
        figsize_overlay: Tuple[float, float] = (9, 4),
        figsize_per_scan: Tuple[float, float] = (8, 4),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> None:
        self.style.apply()
        self._validate_inputs(delays_by_scan, mode=mode, view=view)

        ylabel = f"Delay [{unit}]"
        the_title = title if title is not None else f"Delay distribution - {view}"

        scans_sorted = sorted(int(s) for s in delays_by_scan.keys())

        if mode == "overlay":
            fig, ax = plt.subplots(1, 1, figsize=figsize_overlay)
            ax.set_title(the_title, fontsize=self.style.title_fontsize)
            ax.set_ylabel(ylabel)
            ax.grid()

            label_to_artists: Dict[str, List[object]] = {}

            if view == "scatter":
                ax.set_xlabel("shot index (filtered)")
                for scan in scans_sorted:
                    d = np.asarray(delays_by_scan[scan])
                    if d.size == 0:
                        continue

                    lab = f"{scan} (med {np.median(d):.2f})"
                    line = ax.plot(d, "o", markersize=PlotStyle.marker_size, alpha=alpha, label=lab)[0]

                    targets: List[object] = [line]
                    if show_median:
                        med_line = ax.axhline(np.median(d), alpha=0.2)
                        targets.append(med_line)

                    label_to_artists[lab] = targets

            else:
                ax.set_xlabel(ylabel)
                all_list = [np.asarray(delays_by_scan[s]) for s in scans_sorted if np.asarray(delays_by_scan[s]).size > 0]
                if len(all_list) == 0:
                    plt.show()
                    return
                all_d = np.concatenate(all_list)
                edges = np.linspace(float(all_d.min()), float(all_d.max()), int(bins) + 1)

                for scan in scans_sorted:
                    d = np.asarray(delays_by_scan[scan])
                    if d.size == 0:
                        continue
                    _, _, patches = ax.hist(d, bins=edges, histtype="step", alpha=0.9, label=str(scan))
                    label_to_artists[str(scan)] = list(patches)

                if show_median:
                    gmed = float(np.median(all_d))
                    lab = f"global median {gmed:.2f}"
                    vline = ax.axvline(gmed, alpha=0.4, linestyle="--", label=lab)
                    label_to_artists[lab] = [vline]

            leg = ax.legend(fontsize=9)
            _make_legend_clickable(ax, legend=leg, label_to_artists=label_to_artists)

            plt.tight_layout()

            if save:
                if save_dir is None:
                    raise ValueError("DelayDistributionPlotter.plot(save=True) requires save_dir=...")
                save_figure(
                    fig,
                    save_dir=save_dir,
                    save_name=(save_name or title or f"delay_distribution_{mode}_{view}"),
                    fmt=save_format,
                    dpi=save_dpi,
                    overwrite=save_overwrite,
                )

            plt.show()
            return

        # per_scan
        for scan in scans_sorted:
            d = np.asarray(delays_by_scan[scan])
            if d.size == 0:
                continue

            fig, ax = plt.subplots(1, 1, figsize=figsize_per_scan)
            ax.set_title(f"{the_title} - scan {scan}", fontsize=self.style.title_fontsize)
            ax.set_ylabel(ylabel)
            ax.grid()

            if view == "scatter":
                ax.set_xlabel("shot index (filtered)")
                ax.plot(d, "o", markersize=PlotStyle.marker_size, ms=ms, alpha=alpha)
                if show_median:
                    lab = f"median {np.median(d):.2f}"
                    med_line = ax.axhline(np.median(d), linestyle="--", alpha=0.6, label=lab)
                    leg = ax.legend()
                    _make_legend_clickable(ax, legend=leg, label_to_artists={lab: [med_line]})
            else:
                ax.set_xlabel(ylabel)
                _, _, patches = ax.hist(d, bins=bins, histtype="stepfilled", alpha=0.7, label="hist")
                label_to_artists = {"hist": list(patches)}
                if show_median:
                    lab = f"median {np.median(d):.2f}"
                    vline = ax.axvline(np.median(d), linestyle="--", alpha=0.8, label=lab)
                    label_to_artists[lab] = [vline]
                    leg = ax.legend()
                    _make_legend_clickable(ax, legend=leg, label_to_artists=label_to_artists)

            plt.tight_layout()

            if save:
                if save_dir is None:
                    raise ValueError("DelayDistributionPlotter.plot(save=True) requires save_dir=...")
                save_figure(
                    fig,
                    save_dir=save_dir,
                    save_name=(save_name or title or "delay_distribution"),
                    fmt=save_format,
                    dpi=save_dpi,
                    overwrite=save_overwrite,
                )

            plt.show()

# ============================================================
# 1D pattern plotting
# ============================================================

class Pattern1DPlotter:
    """
    Plots for 1D q/I patterns: overlays + comparison to reference (absolute + differential).
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        self.style = style

    def plot_caked_patterns(
        self,
        patterns: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        *,
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (6, 6),
        legend_ncol: int = 1,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        label_to_artists: Dict[str, List[object]] = {}
        for lab, q, I in patterns:
            lab_s = str(lab)
            line = ax.plot(np.asarray(q), np.asarray(I), label=lab_s, markersize=PlotStyle.marker_size)[0]
            label_to_artists[lab_s] = [line]

        ax.set_xlabel("q [Å$^{-1}$]", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Intensity [a.u.]", fontsize=self.style.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=self.style.title_fontsize)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid()

        leg = ax.legend(ncol=int(legend_ncol), framealpha=1.0, fontsize=self.style.overall_fontsize)
        _make_legend_clickable(ax, legend=leg, label_to_artists=label_to_artists)

        plt.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("Pattern1DPlotter.plot_caked_patterns(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "caked_patterns"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax

    def compare_to_reference(
        self,
        *,
        q_ref: np.ndarray,
        I_ref: np.ndarray,
        ref_label: str,
        patterns: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = (1.5, 4.5),
        ylim_top: Optional[Tuple[float, float]] = None,
        ylim_diff: Optional[Tuple[float, float]] = None,
        vlines_peak: Optional[Tuple[float, float]] = None,
        vlines_bckg: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 7),
        legend_title: str = "Series",
        legend_loc: str = "upper left",
        legend_outside: bool = True,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        self.style.apply()
        q_ref = np.asarray(q_ref)
        I_ref = np.asarray(I_ref)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05, "left": 0.1, "bottom": 0.15, "right": 0.6, "top": 0.85},
        )

        if title is not None:
            ax_top.set_title(title, fontsize=self.style.title_fontsize)

        ref_line = ax_top.plot(q_ref, I_ref, linewidth=2.5, color="black", label=str(ref_label), markersize=PlotStyle.marker_size)[0]

        cmap = plt.cm.jet
        n = max(len(patterns), 1)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        label_to_artists: Dict[str, List[object]] = {str(ref_label): [ref_line]}

        for k, (lab, q, I) in enumerate(patterns):
            lab_s = str(lab)
            q = np.asarray(q)
            I = np.asarray(I)

            top_line = ax_top.plot(q, I, linewidth=1.5, label=lab_s, color=colors[k], markersize=PlotStyle.marker_size)[0]
            qg, dI = _diff_on_ref_grid(q_ref, I_ref, q, I)
            bot_line = ax_bot.plot(qg, dI, linewidth=1.2, color=colors[k], markersize=PlotStyle.marker_size)[0]
            label_to_artists[lab_s] = [top_line, bot_line]

        ax_top.set_ylabel("Intensity [a.u.]", fontsize=self.style.label_fontsize)
        ax_bot.set_ylabel(r"$\Delta$I [a.u.]", fontsize=self.style.label_fontsize)
        ax_bot.set_xlabel(r"q [Å$^{-1}$]", fontsize=self.style.label_fontsize)

        ax_top.grid()
        ax_bot.grid()
        ax_bot.axhline(0.0, linewidth=1.0, alpha=0.7)

        if xlim is not None:
            ax_bot.set_xlim(*xlim)
        if ylim_top is not None:
            ax_top.set_ylim(*ylim_top)
        if ylim_diff is not None:
            ax_bot.set_ylim(*ylim_diff)

        ncol = int(len(patterns) / 15) if int(len(patterns) / 15) > 0 else 1

        if legend_outside:
            leg = ax_top.legend(
                title=legend_title,
                title_fontsize=self.style.label_fontsize - 1,
                fontsize=self.style.overall_fontsize,
                loc=legend_loc,
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                framealpha=1.0,
                ncol=ncol,
            )
        else:
            leg = ax_top.legend(
                title=legend_title,
                title_fontsize=self.style.label_fontsize - 1,
                fontsize=self.style.overall_fontsize,
                loc=legend_loc,
                framealpha=1.0,
                ncol=ncol,
            )

        _make_legend_clickable(ax_top, legend=leg, label_to_artists=label_to_artists)

        #plt.tight_layout()

        if (vlines_peak != None) and (vlines_bckg != None):
            for ax in (ax_top, ax_bot):
                for i in (0,-1):
                    ax.axvline(vlines_peak[i], linestyle="--",linewidth=2.0, color="orange")
                    ax.axvline(vlines_bckg[i], linestyle=":",linewidth=1.0, color="black")

        if save:
            if save_dir is None:
                raise ValueError("Pattern1DPlotter.compare_to_reference(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "compare_to_reference"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, (ax_top, ax_bot)

    # ------------------------------------------------------------------
    # NEW: convenience wrapper for fluence series (no behavior change to core)
    # ------------------------------------------------------------------
    def compare_fluence_to_reference(
        self,
        *,
        q_ref: np.ndarray,
        I_ref: np.ndarray,
        ref_label: str,
        patterns: Sequence[Tuple[Union[float, int, str], np.ndarray, np.ndarray]],
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = (1.5, 4.5),
        ylim_top: Optional[Tuple[float, float]] = None,
        ylim_diff: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 7),
        fluence_unit: str = "mJ/cm$^2$",
        sort_by_numeric_fluence: bool = True,
        legend_loc: str = "upper left",
        legend_outside: bool = True,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Fluence-series wrapper around compare_to_reference().

        Input patterns:
          patterns = [(fluence_value, q, I), ...]
        where fluence_value can be float/int/str; it will be stringified for legend.

        If sort_by_numeric_fluence=True, we attempt numeric sorting (robust fallback to original order).
        """

        pats = list(patterns)

        if sort_by_numeric_fluence and len(pats) >= 2:
            def _to_float_safe(v):
                try:
                    return float(v)
                except Exception:
                    try:
                        return float(str(v).replace("p", "."))
                    except Exception:
                        return np.nan

            vals = np.array([_to_float_safe(p[0]) for p in pats], float)
            if np.isfinite(vals).any():
                # keep non-finite at end, preserve relative order among them
                order = np.argsort(np.where(np.isfinite(vals), vals, np.inf), kind="mergesort")
                pats = [pats[i] for i in order.tolist()]

        # Convert legend labels to compact strings
        patterns_labeled: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for fval, q, I in pats:
            try:
                f = float(fval)
                lab = f"{f:g}"
            except Exception:
                lab = str(fval)
            patterns_labeled.append((lab, np.asarray(q), np.asarray(I)))

        legend_title = f"Fluence [{fluence_unit}]"

        return self.compare_to_reference(
            q_ref=np.asarray(q_ref),
            I_ref=np.asarray(I_ref),
            ref_label=str(ref_label),
            patterns=patterns_labeled,
            title=title,
            xlim=xlim,
            ylim_top=ylim_top,
            ylim_diff=ylim_diff,
            figsize=figsize,
            legend_title=legend_title,
            legend_loc=legend_loc,
            legend_outside=legend_outside,
            save=save,
            save_dir=save_dir,
            save_name=save_name,
            save_format=save_format,
            save_dpi=save_dpi,
            save_overwrite=save_overwrite,
        )

# ============================================================
# CSV-backed plots + fit overlay (calibration-style)
# ============================================================

class FitCSVPlotter:
    """
    Plotters consuming a calibration-style peak_fits.csv-like table.

    Expected columns:
      - success (bool)
      - azim_center, azim_range_str
      - q_fit0, q_fit1
      - bg_c0, bg_c1
      - pv_center, pv_sigma, pv_amplitude, pv_fraction
      - pv_height, pv_fwhm, r2
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        self.style = style

    def plot_property_vs_azimuth(
        self,
        csv_path: str,
        *,
        y: str = "pv_center",
        only_success: bool = True,
        title: Optional[str] = None,
        xlim: Tuple[float, float] = (-95, 95),
        xticks: Sequence[float] = (-90, -60, -30, 0, 30, 60, 90),
        figsize: Tuple[float, float] = (5, 5),
        ylim: Optional[Tuple[float, float]] = None,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()
        df = pd.read_csv(csv_path)

        if only_success and ("success" in df.columns):
            df = df[df["success"] == True]  # noqa: E712

        df = df.sort_values("azim_center", na_position="last")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        line = ax.plot(df["azim_center"].values, df[y].values, "-o", markersize=PlotStyle.marker_size, label=str(y))[0]

        ax.set_xlabel("Azimuthal Angle, $\\Phi$ [°]", fontsize=self.style.label_fontsize)

        if y == "pv_center":
            ylabel = "<q$_{110}$> [Å$^{-1}$]"
            if ylim is None:
                ylim = (2.51, 2.55)
        else:
            ylabel = y

        ax.set_ylabel(ylabel, fontsize=self.style.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=self.style.title_fontsize)

        ax.set_xlim(*xlim)
        ax.set_xticks(list(xticks))
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.grid()

        leg = ax.legend(framealpha=1.0, fontsize=self.style.overall_fontsize)
        _make_legend_clickable(ax, legend=leg, label_to_artists={str(y): [line]})

        plt.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("FitCSVPlotter.plot_property_vs_azimuth(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or f"property_vs_azimuth_{y}"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax

    @staticmethod
    def _make_lmfit_model():
        from lmfit.models import PolynomialModel, PseudoVoigtModel
        bg = PolynomialModel(degree=1, prefix="bg_")
        pv = PseudoVoigtModel(prefix="pv_")
        return bg + pv

    @staticmethod
    def _eval_model_from_row(
        q_eval: np.ndarray,
        row,
        *,
        eta_default: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_eval = np.asarray(q_eval)

        model = FitCSVPlotter._make_lmfit_model()
        params = model.make_params()

        params["bg_c0"].set(value=float(row["bg_c0"]))
        params["bg_c1"].set(value=float(row["bg_c1"]))
        params["pv_center"].set(value=float(row["pv_center"]))
        params["pv_sigma"].set(value=float(row["pv_sigma"]))
        params["pv_amplitude"].set(value=float(row["pv_amplitude"]))

        frac = float(row["pv_fraction"]) if ("pv_fraction" in row) else float(eta_default)
        params["pv_fraction"].set(value=frac, vary=False, min=0.0, max=1.0)

        yfit = model.eval(params, x=q_eval)
        comps = model.eval_components(params=params, x=q_eval)
        bg = comps.get("bg_", np.zeros_like(q_eval))
        pv = comps.get("pv_", np.zeros_like(q_eval))
        return np.asarray(yfit, float), np.asarray(bg, float), np.asarray(pv, float)

    def plot_fit_overlay(
        self,
        *,
        q: np.ndarray,
        I: np.ndarray,
        csv_path: str,
        azim_range_str: str,
        title: Optional[str] = None,
        fit_oversample: int = 10,
        eta_default: float = 0.3,
        figsize: Tuple[float, float] = (6, 6),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()

        q = np.asarray(q)
        I = np.asarray(I)

        df = pd.read_csv(csv_path)
        sel = df[df["azim_range_str"] == azim_range_str]
        if len(sel) == 0:
            available = sorted(df["azim_range_str"].astype(str).unique().tolist())
            raise KeyError(
                f"azim_range_str='{azim_range_str}' not found in CSV.\n"
                f"Available: {available}"
            )

        row = sel.iloc[0]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if title is not None:
            ax.set_title(title, fontsize=self.style.title_fontsize)

        scat = ax.scatter(q, I, color="black", s=self.style.marker_size, label="Exp. data")

        if ("success" in row) and (not bool(row["success"])):
            ax.set_xlabel("q [Å$^{-1}$]", fontsize=self.style.label_fontsize)
            ax.set_ylabel("Intensity [a.u.]", fontsize=self.style.label_fontsize)
            ax.grid()
            leg = ax.legend(fontsize=self.style.overall_fontsize)
            _make_legend_clickable(ax, legend=leg, label_to_artists={"Exp. data": [scat]})
            plt.tight_layout()

            if save:
                if save_dir is None:
                    raise ValueError("FitCSVPlotter.plot_fit_overlay(save=True) requires save_dir=...")
                save_figure(
                    fig,
                    save_dir=save_dir,
                    save_name=(save_name or title or f"fit_overlay_{azim_range_str}"),
                    fmt=save_format,
                    dpi=save_dpi,
                    overwrite=save_overwrite,
                )

            plt.show()
            return fig, ax

        q0 = float(row["q_fit0"])
        q1 = float(row["q_fit1"])
        i0 = int(np.argmin(np.abs(q - q0)))
        i1 = int(np.argmin(np.abs(q - q1))) + 1

        q_fit = q[i0:i1]
        if len(q_fit) < 2:
            if save:
                if save_dir is None:
                    raise ValueError("FitCSVPlotter.plot_fit_overlay(save=True) requires save_dir=...")
                save_figure(
                    fig,
                    save_dir=save_dir,
                    save_name=(save_name or title or f"fit_overlay_{azim_range_str}"),
                    fmt=save_format,
                    dpi=save_dpi,
                    overwrite=save_overwrite,
                )
            plt.show()
            return fig, ax

        fit_oversample = max(int(fit_oversample), 1)
        n_dense = max(len(q_fit) * fit_oversample, len(q_fit) + 1)
        q_dense = np.linspace(float(q_fit[0]), float(q_fit[-1]), n_dense)

        yfit, bg, pv = self._eval_model_from_row(q_dense, row, eta_default=eta_default)
        peak_only_shifted = pv + 0.5 * (bg[0] + bg[-1])

        fit_label = (
            "Full fit\n"
            f"I = {float(row.get('pv_height', np.nan)):.3g} a.u.\n"
            f"q = {float(row.get('pv_center', np.nan)):.3f} Å$^{{-1}}$\n"
            f"fwhm = {abs(float(row.get('pv_fwhm', np.nan))):.3g} Å$^{{-1}}$\n"
            f"R$^2$ = {float(row.get('r2', np.nan)):.3f}"
        )

        line_fit = ax.plot(q_dense, yfit, color="orange", linewidth=2, label=fit_label)[0]
        line_pv = ax.plot(q_dense, peak_only_shifted, color="red", linestyle="--", linewidth=1, label="pseudo-Voigt")[0]
        line_bg = ax.plot(q_dense, bg, color="gray", linestyle="--", linewidth=1, label="background")[0]

        ax.set_xlabel("q [Å$^{-1}$]", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Intensity [a.u.]", fontsize=self.style.label_fontsize)
        ax.set_xlim(q_dense[0] * 0.985, q_dense[-1] * 1.045)
        ax.set_ylim(float(np.min(yfit)) * 0.9, float(np.max(yfit)) * 1.2)

        ax.grid()
        leg = ax.legend(fontsize=self.style.overall_fontsize, loc="upper right")

        label_to_artists = {
            "Exp. data": [scat],
            fit_label: [line_fit],
            "pseudo-Voigt": [line_pv],
            "background": [line_bg],
        }
        _make_legend_clickable(ax, legend=leg, label_to_artists=label_to_artists)

        plt.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("FitCSVPlotter.plot_fit_overlay(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or f"fit_overlay_{azim_range_str}"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax


# ============================================================
# Peak fit overlay (used by fitting_utils.py / fitting.py)
# ============================================================

class PeakFitOverlayPlotter:
    """
    Calibration-like 1D + fit overlay.

    Keep this class name and public methods stable; other scripts import it.
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        self.style = style

    def plot_from_payload(
        self,
        payload: dict,
        *,
        title: str,
        show: bool = False,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: str = "peak_fit_overlay",
        save_format: str = "png",
        save_dpi: int = 300,
        save_overwrite: bool = True,
        close_after: bool = True,
    ):
        """
        Convenience wrapper:
        - during fitting (payload produced by fit_one_peak)
        - after fitting (payload reconstructed from CSV)
        """
        if (not payload) or (not bool(payload.get("success", False))):
            return None, None, None

        fit_label = (
            "Full fit\n"
            f"I = {float(payload.get('pv_height', np.nan)):.3g} a.u.\n"
            f"q = {float(payload.get('pv_center', np.nan)):.3f} Å$^{{-1}}$\n"
            f"fwhm = {abs(float(payload.get('pv_fwhm', np.nan))):.3g} Å$^{{-1}}$\n"
            f"R$^2$ = {float(payload.get('r2', np.nan)):.3f}"
        )

        return self.plot(
            q=payload["q"],
            I=payload["I"],
            q_dense=payload["q_dense"],
            yfit=payload["y_dense"],
            bg=payload["bg_dense"],
            pv=payload["pv_dense"],
            fit_label=fit_label,
            title=title,
            show=bool(show),
            save=bool(save),
            save_dir=save_dir,
            save_name=str(save_name),
            save_format=str(save_format),
            save_dpi=int(save_dpi),
            save_overwrite=bool(save_overwrite),
            close_after=bool(close_after),
        )

    def plot(
        self,
        *,
        q: np.ndarray,
        I: np.ndarray,
        q_dense: Optional[np.ndarray] = None,
        yfit: Optional[np.ndarray] = None,
        bg: Optional[np.ndarray] = None,
        pv: Optional[np.ndarray] = None,
        fit_label: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (6, 6),
        xlim_pad_left: float = 0.985,
        xlim_pad_right: float = 1.045,
        ylim_pad_low: float = 0.9,
        ylim_pad_high: float = 1.2,
        show: bool = True,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 300,
        save_overwrite: bool = True,
        close_after: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes, Optional[str]]:
        self.style.apply()

        q = np.asarray(q, float)
        I = np.asarray(I, float)

        qd = None if q_dense is None else np.asarray(q_dense, float)
        yf = None if yfit is None else np.asarray(yfit, float)
        b = None if bg is None else np.asarray(bg, float)
        p = None if pv is None else np.asarray(pv, float)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        scat = ax.scatter(q, I, color="black", s=float(self.style.marker_size), label="Exp. data")

        line_fit = None
        line_pv = None
        line_bg = None

        if (qd is not None) and (yf is not None) and (b is not None) and (p is not None):
            peak_only_shifted = p + 0.5 * (float(b[0]) + float(b[-1]))

            if fit_label is None or str(fit_label).strip() == "":
                fit_label = "Full fit"

            line_fit = ax.plot(qd, yf, color="orange", linewidth=2, label=str(fit_label))[0]
            line_pv = ax.plot(qd, peak_only_shifted, color="red", linestyle="--", linewidth=1, label="pseudo-Voigt")[0]
            line_bg = ax.plot(qd, b, color="gray", linestyle="--", linewidth=1, label="background")[0]

            if qd.size >= 2:
                ax.set_xlim(float(qd[0]) * float(xlim_pad_left), float(qd[-1]) * float(xlim_pad_right))
            if yf.size >= 1 and np.isfinite(yf).any():
                ymin = float(np.nanmin(yf)) * float(ylim_pad_low)
                ymax = float(np.nanmax(yf)) * float(ylim_pad_high)
                if np.isfinite(ymin) and np.isfinite(ymax) and (ymax > ymin):
                    ax.set_ylim(ymin, ymax)
        else:
            if q.size >= 2:
                ax.set_xlim(float(q[0]) * float(xlim_pad_left), float(q[-1]) * float(xlim_pad_right))
            if I.size >= 1 and np.isfinite(I).any():
                ymin = float(np.nanmin(I)) * float(ylim_pad_low)
                ymax = float(np.nanmax(I)) * float(ylim_pad_high)
                if np.isfinite(ymin) and np.isfinite(ymax) and (ymax > ymin):
                    ax.set_ylim(ymin, ymax)

        ax.set_xlabel("q [Å$^{-1}$]", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Intensity [a.u.]", fontsize=self.style.label_fontsize)
        ax.grid()

        if title is not None and str(title).strip() != "":
            fig.suptitle(str(title), fontsize=self.style.title_fontsize, y=0.98)

        label_to_artists: Dict[str, List[object]] = {"Exp. data": [scat]}
        if line_fit is not None and fit_label is not None:
            label_to_artists[str(fit_label)] = [line_fit]
        if line_pv is not None:
            label_to_artists["pseudo-Voigt"] = [line_pv]
        if line_bg is not None:
            label_to_artists["background"] = [line_bg]

        leg = ax.legend(fontsize=self.style.overall_fontsize, loc="upper right", framealpha=1.0)
        _make_legend_clickable(ax, legend=leg, label_to_artists=label_to_artists)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        saved_path: Optional[str] = None
        if bool(save):
            if save_dir is None:
                raise ValueError("PeakFitOverlayPlotter.plot(save=True) requires save_dir=...")

            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            fmt = str(save_format).lstrip(".").lower() or "png"
            stem = _sanitize_stem(save_name or "peak_fit_overlay")
            out_path = out_dir / f"{stem}.{fmt}"

            # Preserve previous behavior: if overwrite=False and file exists, do not create a new suffix.
            if out_path.exists() and (not bool(save_overwrite)):
                saved_path = str(out_path)
            else:
                out_path = save_figure(
                    fig,
                    save_dir=out_dir,
                    save_name=stem,
                    fmt=fmt,
                    dpi=int(save_dpi),
                    overwrite=True,
                )
                saved_path = str(out_path)

        if bool(show):
            plt.show()
        if bool(close_after):
            try:
                plt.close(fig)
            except Exception:
                pass

        return fig, ax, saved_path


def default_label_from_experiment_multi(exp: dict) -> str:
    # No units here; units are declared in legend title.
    sn = str(exp.get("sample_name", ""))
    try:
        tK = int(exp.get("temperature_K", 0))
    except Exception:
        tK = 0
    try:
        wl = int(float(exp.get("excitation_wl_nm", 0)))
    except Exception:
        wl = 0
    try:
        flu = float(exp.get("fluence_mJ_cm2", 0.0))
    except Exception:
        flu = 0.0
    try:
        tw = int(exp.get("time_window_fs", 0))
    except Exception:
        tw = 0

    return f"{sn}, {tK}, {wl}, {flu:g}, {tw}"


def legend_title_default(scan_type) -> str:
    # Units live in the legend title (not in each label)
    if scan_type.lower() == "fluence":
        return "Sample, T [K], $\\lambda_{ex}$ [nm], delay [fs], time bin [fs]"
    
    elif scan_type.lower() == "delay":
        return "Sample, T [K], $\\lambda_{ex}$ [nm], flu [mJ/cm$^2$], time bin [fs]"


class FitTimeEvolutionPlotter:
    """
    Plot evolution of a fitted property for scans.

    Backwards-compatible default: delay scans (x_col='delay_fs', unit='ps'/'fs').

    New capability:
      - can plot vs any numeric column (e.g. fluence_mJ_cm2) by passing x_col=...
      - for non-delay x_col, 'unit' is treated as a display label only (no scaling),
        unless you explicitly provide x_scale.
    """

    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = DEFAULT_STYLE if style is None else style

    @staticmethod
    def _canonical_group_keys(
        groups,
        *,
        group_by: str,
    ) -> Optional[List[str]]:
        if groups is None:
            return None

        keep_keys: List[str] = []
        if str(group_by) == "azim_range_str":
            for g in list(groups):
                if isinstance(g, (list, tuple)) and len(g) == 2:
                    keep_keys.append(general_utils.azim_range_str((float(g[0]), float(g[1]))))
                else:
                    keep_keys.append(str(g))
        else:
            keep_keys = [str(g) for g in list(groups)]

        return keep_keys

    @staticmethod
    def _ylabel_for(peak: str, y: str) -> str:
        pk = str(peak)
        yy = str(y)

        if yy == "hkl_pos":
            return rf"<q$_{{{pk}}}$> [Å$^{{-1}}$]"
        if yy == "hkl_fwhm":
            return rf"FWHM$_{{{pk}}}$ [Å$^{{-1}}$]"
        if yy in ("hkl_i", "hkl_intensity"):
            return rf"I$_{{{pk}}}$ [a.u.]"
        if yy in ("hkl_height",):
            return rf"Height$_{{{pk}}}$ [a.u.]"
        if yy in ("r2", "R2", "R^2", "r_squared"):
            return r"R$^2$"
        return yy

    @staticmethod
    def _robust_sigma_mad(resid: np.ndarray) -> float:
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            return np.nan
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        return 1.4826 * mad

    @staticmethod
    def _is_full_window_row(row_phi0: float, row_phi1: float, *, tol: float = 1e-6) -> bool:
        if not (np.isfinite(row_phi0) and np.isfinite(row_phi1)):
            return False
        w = abs(float(row_phi1) - float(row_phi0))
        a0 = abs(float(row_phi0))
        a1 = abs(float(row_phi1))
        if abs(w - 180.0) <= tol:
            return True
        if (abs(a0 - 90.0) <= tol) and (abs(a1 - 90.0) <= tol):
            return True
        return False

    @staticmethod
    def _infer_phi_halfwidth_deg(
        dframe,
        *,
        group_by: str,
    ) -> Optional[float]:
        if dframe is None or len(dframe) == 0:
            return None

        df = dframe.copy()

        if group_by in df.columns and str(group_by) == "phi_label":
            df = df[df[group_by].astype(str) != "Full"].copy()

        if "azim_range_str" in df.columns:
            df = df[df["azim_range_str"].astype(str) != "-90_90"].copy()

        if ("phi0" in df.columns) and ("phi1" in df.columns):
            a = df["phi0"].astype(float).values
            b = df["phi1"].astype(float).values
            m = np.isfinite(a) & np.isfinite(b)
            if np.any(m):
                full_mask = np.zeros(len(df), dtype=bool)
                idx = np.where(m)[0]
                for ii in idx:
                    if FitTimeEvolutionPlotter._is_full_window_row(a[ii], b[ii]):
                        full_mask[ii] = True
                df = df[~full_mask].copy()

        if len(df) == 0:
            return None

        if "phi_halfwidth_deg" in df.columns:
            hw = df["phi_halfwidth_deg"].astype(float).values
            hw = hw[np.isfinite(hw)]
            hw = hw[(hw > 0) & (np.abs(hw - 90.0) > 1e-6)]
        elif ("phi0" in df.columns) and ("phi1" in df.columns):
            a = df["phi0"].astype(float).values
            b = df["phi1"].astype(float).values
            m = np.isfinite(a) & np.isfinite(b)
            if not np.any(m):
                return None
            hw = 0.5 * np.abs(b[m] - a[m])
            hw = hw[(hw > 0) & (np.abs(hw - 90.0) > 1e-6)]
        else:
            return None

        if hw.size == 0:
            return None

        hw_r = np.round(hw.astype(float), 6)
        vals, counts = np.unique(hw_r, return_counts=True)
        if vals.size == 1:
            return float(vals[0])
        if counts.max() == 1:
            return float(np.min(vals))
        return float(vals[int(np.argmax(counts))])

    def plot(
        self,
        df,
        *,
        peak: str,
        y: str,
        unit: str = "ps",
        group_by: str = "azim_range_str",
        groups: Optional[Sequence[Union[str, float, int, Tuple[float, float]]]] = None,
        only_success: bool = True,
        include_reference: bool = True,
        title: Optional[str] = None,
        as_lines: bool = False,
        delay_offset: float = 0.0,
        figsize: Tuple[float, float] = (7, 6),
        alpha: float = 0.85,
        ms: float = 4.0,
        ref_alpha: float = 0.5,
        ref_linestyle: str = ":",
        # ---- baseline uncertainty
        show_baseline_sigma: bool = False,
        baseline_sigma: float = 1.0,
        baseline_alpha: float = 1.0,
        baseline_mode: str = "errorbar",
        baseline_estimator: str = "std",
        baseline_ddof: int = 1,
        # ---- legend controls
        legend_title: Optional[str] = None,
        legend_loc: str = "upper left",
        legend_outside: bool = True,
        legend_bbox: Tuple[float, float] = (1.02, 1.0),
        # ---- explicit margins
        left: float = 0.20,
        bottom: float = 0.20,
        top: float = 0.8,
        right_inside: float = 0.96,
        right_outside: float = 0.70,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
        # ---- NEW (backwards compatible)
        x_col: str = "delay_fs",
        x_label: Optional[str] = None,
        x_scale: Optional[float] = None,
        x_offset: float = 0.0,
    ):
        if self.style is not None and hasattr(self.style, "apply"):
            self.style.apply()

        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe.")
        if "peak" not in df.columns:
            raise KeyError("DataFrame missing required column: 'peak'")
        if group_by not in df.columns:
            raise KeyError(f"DataFrame missing required column: '{group_by}'")
        if y not in df.columns:
            raise KeyError(f"DataFrame missing requested y column: '{y}'")

        x_col = str(x_col)
        if x_col not in df.columns:
            raise KeyError(f"DataFrame missing required x column: '{x_col}'")

        d0 = df[df["peak"].astype(str) == str(peak)].copy()

        if only_success and ("success" in d0.columns):
            d0 = d0[d0["success"] == True].copy()  # noqa: E712

        if include_reference and ("is_reference" in d0.columns):
            d_ref = d0[d0["is_reference"] == True].copy()  # noqa: E712
            d_dat = d0[d0["is_reference"] != True].copy()  # noqa: E712
        else:
            d_ref = d0.iloc[0:0].copy()
            d_dat = d0.copy()

        # finite x only
        d_dat = d_dat[np.isfinite(d_dat[x_col].astype(float).values)].copy()

        keep_keys = self._canonical_group_keys(groups, group_by=str(group_by))

        if keep_keys is not None:
            d_dat = d_dat[d_dat[group_by].astype(str).isin(keep_keys)].copy()
            d_ref = d_ref[d_ref[group_by].astype(str).isin(keep_keys)].copy()

        if len(d_dat) == 0 and len(d_ref) == 0:
            raise ValueError("No rows to plot after filtering (peak/groups/only_success).")

        # ---- x axis behavior
        # Default (delay): scale by fs/ps and use delay_offset.
        # Non-delay: no scaling unless x_scale is given; use x_offset.
        if x_col == "delay_fs":
            unit_l = str(unit).strip().lower()
            if unit_l not in ("fs", "ps"):
                raise ValueError("unit must be 'fs' or 'ps' when x_col='delay_fs'")
            scale = 1.0 if unit_l == "fs" else 1e-3
            x_off = float(delay_offset)
            xlab = x_label if x_label is not None else f"Delay [{unit_l}]"
        else:
            scale = float(x_scale) if x_scale is not None else 1.0
            x_off = float(x_offset)
            xlab = x_label if x_label is not None else str(x_col)

        if keep_keys is None:
            keep_keys = sorted(d_dat[group_by].astype(str).unique().tolist())
        keep_keys = list(keep_keys)

        ngrp = max(len(keep_keys), 1)
        cmap = plt.cm.jet
        colors = [cmap(i / max(ngrp - 1, 1)) for i in range(ngrp)]
        color_map = {g: colors[i] for i, g in enumerate(keep_keys)}

        # preserve  special-casing
        if str(group_by) == "azim_range_str":
            full_key = general_utils.azim_range_str((-90.0, 90.0))
            if full_key in color_map:
                color_map[full_key] = "black"
        if "Full" in color_map:
            color_map["Full"] = "black"

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        r = float(right_outside) if legend_outside else float(right_inside)
        fig.subplots_adjust(left=float(left), bottom=float(bottom), top=float(top), right=r)

        title_fs = PlotStyle.title_fontsize
        label_fs = PlotStyle.label_fontsize
        leg_fs = PlotStyle.overall_fontsize
        marker_size = PlotStyle.marker_size

        if title is not None:
            ax.set_title(str(title), fontsize=title_fs)

        ax.set_xlabel(str(xlab), fontsize=label_fs)
        ax.set_ylabel(self._ylabel_for(str(peak), str(y)), fontsize=label_fs)
        ax.tick_params(axis="both", which="both")
        ax.grid()

        # x range for baseline band
        x_min, x_max = -1.0, 1.0
        if len(d_dat) > 0:
            xx = d_dat[x_col].astype(float).values * float(scale) + float(x_off)
            xx = xx[np.isfinite(xx)]
            if xx.size > 0:
                x_min, x_max = float(np.min(xx)), float(np.max(xx))

        label_to_artists: Dict[str, List[object]] = {}

        labels_for_legend = list(groups) if groups is not None else list(keep_keys)

        for k, g in enumerate(keep_keys):
            cg = color_map.get(g, None)
            dg = d_dat[d_dat[group_by].astype(str) == str(g)].copy()
            artists: List[object] = []

            yref = None
            if include_reference and len(d_ref) > 0:
                rg = d_ref[d_ref[group_by].astype(str) == str(g)].copy()
                if len(rg) > 0:
                    vals = rg[y].astype(float).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        yref = float(vals[0])
                        hline = ax.axhline(
                            yref,
                            color=cg,
                            alpha=float(ref_alpha),
                            linestyle=str(ref_linestyle),
                            linewidth=2.0,
                        )
                        artists.append(hline)

            if len(dg) > 0:
                x = dg[x_col].astype(float).values * float(scale) + float(x_off)
                yy = dg[y].astype(float).values

                tw = None
                if (x_col == "delay_fs") and ("time_window_fs" in dg.columns):
                    tw = pd.to_numeric(dg["time_window_fs"], errors="coerce").values.astype(float)

                ok = np.isfinite(x) & np.isfinite(yy)
                x = x[ok]
                yy = yy[ok]
                if tw is not None:
                    tw = tw[ok]

                order = np.argsort(x)
                x = x[order]
                yy = yy[order]
                if tw is not None:
                    tw = tw[order]
                    xerr = 0.5 * tw * float(scale)
                    xerr = np.asarray(xerr, float)
                    xerr[~np.isfinite(xerr)] = 0.0
                    if not np.any(xerr > 0):
                        xerr = None
                else:
                    xerr = None

                lab = str(labels_for_legend[k])

                if as_lines:
                    line = ax.plot(
                        x, yy, "-o", markersize=PlotStyle.marker_size,
                        #ms=float(ms),
                        alpha=float(alpha),
                        color=cg,
                        label=lab,
                    )[0]
                    artists.append(line)
                else:
                    sc = ax.scatter(
                        x, yy,
                        s=marker_size,
                        alpha=float(alpha),
                        color=cg,
                        label=lab,
                    )
                    artists.append(sc)

                if xerr is not None:
                    ebx = ax.errorbar(
                        x, yy,
                        xerr=xerr,
                        fmt="none",
                        ecolor=cg,
                        elinewidth=1.2,
                        alpha=0.35,
                        capsize=0,
                    )
                    artists.extend(_flatten_errorbar_container(ebx))

                # ---- baseline sigma logic (unchanged):
                # uses x < 0 as the baseline region.
                if show_baseline_sigma and (yref is not None):
                    neg_mask = x < 0.0
                    if np.any(neg_mask):
                        resid = yy[neg_mask] - float(yref)
                        resid = resid[np.isfinite(resid)]

                        sigma = np.nan
                        if resid.size >= 2:
                            est = str(baseline_estimator).lower()
                            if est == "mad":
                                sigma = float(self._robust_sigma_mad(resid))
                            else:
                                sigma = float(np.std(resid, ddof=int(baseline_ddof)))
                        elif resid.size == 1:
                            sigma = 0.0

                        if np.isfinite(sigma) and sigma > 0:
                            band = float(baseline_sigma) * float(sigma)
                            mode = str(baseline_mode).lower()

                            if mode == "errorbar":
                                eb = ax.errorbar(
                                    x, yy,
                                    yerr=band,
                                    fmt="none",
                                    ecolor=cg,
                                    elinewidth=1.2,
                                    alpha=0.4,#float(baseline_alpha),
                                    capsize=0,
                                )
                                artists.extend(_flatten_errorbar_container(eb))
                            else:
                                poly = ax.fill_between(
                                    [x_min, x_max],
                                    [yref - band, yref - band],
                                    [yref + band, yref + band],
                                    color=cg,
                                    alpha=float(baseline_alpha),
                                    linewidth=0.0,
                                    zorder=0,
                                )
                                artists.append(poly)

            if artists:
                label_to_artists[str(labels_for_legend[k])] = artists

        phi_avg_context = False
        if str(group_by) in ("phi_label", "phi_center_abs"):
            phi_avg_context = True
        elif "phi_mode" in d0.columns:
            try:
                phi_avg_context = str(d0["phi_mode"].dropna().iloc[0]) == "phi_avg"
            except Exception:
                phi_avg_context = False

        if legend_title is None:
            if phi_avg_context:
                hw = self._infer_phi_halfwidth_deg(d0, group_by=str(group_by))
                if hw is not None and np.isfinite(hw):
                    hw_s = str(int(round(hw))) if abs(hw - round(hw)) < 1e-9 else f"{hw:g}"
                    leg_title = rf"$|\Phi| \pm {hw_s}$ [°]"
                else:
                    leg_title = r"$|\Phi|$ [°]"
            else:
                leg_title = "Azim. Range [°]\n($\\Phi_{o}$,$\\Phi_{f}$)"
        else:
            leg_title = str(legend_title)

        if legend_outside:
            leg = ax.legend(
                title=leg_title,
                title_fontsize=label_fs - 1,
                fontsize=leg_fs,
                loc=str(legend_loc),
                bbox_to_anchor=tuple(legend_bbox),
                borderaxespad=0.0,
                framealpha=1.0,
            )
        else:
            leg = ax.legend(
                title=leg_title,
                title_fontsize=label_fs - 1,
                fontsize=leg_fs,
                loc=str(legend_loc),
                framealpha=1.0,
            )

        t = leg.get_title()
        t.set_multialignment("center")
        t.set_ha("center")

        _make_legend_clickable(ax, legend=leg, label_to_artists=label_to_artists)

        if save:
            if save_dir is None:
                raise ValueError("FitTimeEvolutionPlotter.plot(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or f"fit_evolution_{x_col}_{peak}_{y}"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax


class FitTimeEvolutionMultiPlotter:
    """
    Plot multiple experiments on the same axes (time evolution).

    Notes:
      - Legend labels are clickable: click a legend entry to toggle that series.
      - If show_baseline_sigma=True and baseline_mode="errorbar":
          one constant yerr is applied at EACH time point of a series
          (computed upstream and passed via series dict as "baseline_sig").
      - Supports merged series naturally:
          * xerr can vary point-by-point
          * baseline_sig remains common per plotted series
      - If save=True and save_dir is None, saves into:
          .../general_figures/
    """

    def __init__(self, style=None, paths: AnalysisPaths | None = None):
        self.style = style
        self.paths = paths
        
    # ----------------------------
    # Defaults / naming / labels
    # ----------------------------
    def _default_general_figures_dir(self) -> str:
        if self.paths is None:
            raise ValueError("A paths object is required to build default save directories.")
        return str(self.paths.analysis_root / "general_figures")

    @staticmethod
    def legend_title_default() -> str:
        legend_title = legend_title_default(scan_type="delay")
        return legend_title

    @staticmethod
    def default_label_from_experiment(exp: dict) -> str:
        label = default_label_from_experiment_multi(exp)
        return label

    @staticmethod
    def ylabel_for_property(prop: str, peak: str) -> str:
        p = str(prop).strip()
        pk = str(peak).strip()
        peak_tex = "$_{" + pk + "}$"

        if p == "hkl_pos":
            return f"<q{peak_tex}> [Å$^{{-1}}$]"
        if p == "hkl_fwhm":
            return f"FWHM{peak_tex} [Å$^{{-1}}$]"
        if p == "hkl_i":
            return f"I{peak_tex} [a.u.]"
        if p == "hkl_area":
            return f"Area{peak_tex} [a.u.]"

        return str(prop)

    @staticmethod
    def title_default(*, peak: str, prop: str, group_by: str = "", group_key: str = "") -> str:
        t = f"hkl=({str(peak)}), {str(prop)}"
        gb = str(group_by).strip()
        gk = str(group_key).strip()
        if gb and gk:
            t += f"\n{gb}={gk}"
        return t

    @staticmethod
    def _sanitize_token(s: str) -> str:
        s = str(s)
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("_", "-", "."):
                out.append(ch)
            else:
                out.append("_")
        tok = "".join(out)
        while "__" in tok:
            tok = tok.replace("__", "_")
        return tok.strip("_")

    @classmethod
    def default_save_name(
        cls,
        *,
        peak: str,
        prop: str,
        group_by: str = "",
        group_key: str = "",
        phi_mode: str = "",
        phi_reduce: str = "",
        n_series: int = 0,
        unit: str = "",
    ) -> str:
        gb = cls._sanitize_token(str(group_by)) if group_by else ""
        gk = cls._sanitize_token(str(group_key)) if group_key else ""
        grp_tok = (gb + "_" + gk).strip("_") if (gb or gk) else "group"
        mode_tok = cls._sanitize_token(str(phi_mode)) if phi_mode else "mode"
        red_tok = cls._sanitize_token(str(phi_reduce)) if phi_reduce else ""
        u = cls._sanitize_token(str(unit)) if unit else ""

        base = f"compare_delay_peak{cls._sanitize_token(peak)}_{cls._sanitize_token(prop)}_{grp_tok}"
        if mode_tok:
            base += f"_{mode_tok}"
        if red_tok:
            base += f"_{red_tok}"
        base += f"_N{int(n_series)}"
        if u:
            base += f"_{u}"
        return base

    # ----------------------------
    # Style
    # ----------------------------
    def _apply_style(self):
        s = self.style
        if s is None:
            return
        try:
            if hasattr(s, "apply") and callable(getattr(s, "apply")):
                s.apply()
                return
        except Exception:
            pass
        try:
            if callable(s):
                s()
                return
        except Exception:
            pass
        try:
            if isinstance(s, dict):
                plt.rcParams.update(dict(s))
        except Exception:
            pass

    # ----------------------------
    # Core helpers
    # ----------------------------
    @staticmethod
    def delay_offset_in_unit(exp: dict, unit: str, global_override=None) -> float:
        """
        Convert per-experiment offset to the requested unit.

        Preferred keys:
          - delay_offset_ps / delay_offset_fs
        Fallback:
          - delay_offset (assumed already in *unit*)
        """
        if global_override is not None:
            try:
                return float(global_override)
            except Exception:
                return 0.0

        u = str(unit).strip().lower()
        if u not in ("fs", "ps"):
            u = "ps"

        if u == "ps":
            if exp.get("delay_offset_ps", None) is not None:
                return float(exp["delay_offset_ps"])
            if exp.get("delay_offset_fs", None) is not None:
                return float(exp["delay_offset_fs"]) * 1e-3
        else:
            if exp.get("delay_offset_fs", None) is not None:
                return float(exp["delay_offset_fs"])
            if exp.get("delay_offset_ps", None) is not None:
                return float(exp["delay_offset_ps"]) * 1e3

        if exp.get("delay_offset", None) is not None:
            try:
                return float(exp["delay_offset"])
            except Exception:
                return 0.0

        return 0.0

    @staticmethod
    def _robust_sigma(y: np.ndarray, *, estimator: str = "std", ddof: int = 1) -> float:
        yy = np.asarray(y, float)
        yy = yy[np.isfinite(yy)]
        if yy.size < 2:
            return float("nan")

        est = str(estimator).strip().lower()
        if est == "mad":
            med = float(np.nanmedian(yy))
            mad = float(np.nanmedian(np.abs(yy - med)))
            return float(1.4826 * mad)

        try:
            return float(np.nanstd(yy, ddof=int(ddof)))
        except Exception:
            return float(np.nanstd(yy))

    @classmethod
    def baseline_from_negative_delay(
        cls,
        *,
        x: np.ndarray,
        y: np.ndarray,
        sigma_scale: float = 1.0,
        estimator: str = "std",
        ddof: int = 1,
    ):
        """
        Compute baseline from the negative-delay region only.

        Uses:
          y0  = mean(y[x < 0])
          sig = std(y[x < 0] - y0)    or MAD-equivalent if estimator='mad'

        This is the intended behavior for merged short/long delay traces when
        one common vertical error bar should describe the full merged series.
        """
        xx = np.asarray(x, float)
        yy = np.asarray(y, float)

        mneg = np.isfinite(xx) & np.isfinite(yy) & (xx < 0.0)
        yneg = yy[mneg]

        if yneg.size == 0:
            return float("nan"), float("nan")

        y0 = float(np.mean(yneg))

        if yneg.size >= 2:
            resid = yneg - y0
            est = str(estimator).strip().lower()
            if est == "mad":
                med = float(np.median(resid))
                mad = float(np.median(np.abs(resid - med)))
                sig = float(1.4826 * mad)
            else:
                sig = float(np.std(resid, ddof=int(ddof)))
        else:
            sig = 0.0

        return float(y0), float(sigma_scale) * float(sig)

    @classmethod
    def baseline_from_reference(
        cls,
        *,
        df_sel: "pd.DataFrame",
        x: np.ndarray,
        y: np.ndarray,
        prop: str,
        cols,
        ref_type: str = "dark",
        ref_value=None,
        sigma_scale: float = 1.0,
        estimator: str = "std",
        ddof: int = 1,
    ):
        """
        Compute (baseline_y0, baseline_sig).

        Priority:
          1) If df_sel has columns "baseline_y0"/"baseline_sig" (or "baseline_sigma"): use them
          2) If ref_type is "negative_delay"/"merged_negative_delay": use x<0 region (mean-centered)
          3) If ref_type == "dark" and cols.is_ref_col exists: use rows where is_ref==True
          4) If ref_type == "delay" and ref_value provided: use rows matching delay_fs in ref_value
          5) Fallback: use x<0 region (mean-centered)
        """
        for sig_col in ("baseline_sig", "baseline_sigma"):
            if sig_col in df_sel.columns:
                try:
                    sigv = float(pd.to_numeric(df_sel[sig_col], errors="coerce").dropna().iloc[0])
                except Exception:
                    sigv = float("nan")
                y0v = float("nan")
                if "baseline_y0" in df_sel.columns:
                    try:
                        y0v = float(pd.to_numeric(df_sel["baseline_y0"], errors="coerce").dropna().iloc[0])
                    except Exception:
                        y0v = float("nan")
                return y0v, float(sigma_scale) * sigv

        ref_type_s = str(ref_type).strip().lower()

        xx = np.asarray(x, float)
        yy = np.asarray(y, float)

        if ref_type_s in ("negative_delay", "merged_negative_delay", "neg_delay", "negative"):
            return cls.baseline_from_negative_delay(
                x=xx,
                y=yy,
                sigma_scale=float(sigma_scale),
                estimator=str(estimator),
                ddof=int(ddof),
            )

        if ref_type_s == "dark":
            if hasattr(cols, "is_ref_col") and cols.is_ref_col in df_sel.columns:
                try:
                    mref = df_sel[cols.is_ref_col].astype(bool).values
                    yref = pd.to_numeric(df_sel[str(prop)], errors="coerce").values.astype(float)
                    yref = yref[np.isfinite(yref) & mref]
                    if yref.size >= 2:
                        y0 = float(np.nanmedian(yref))
                        sig = cls._robust_sigma(yref, estimator=estimator, ddof=ddof)
                        return y0, float(sigma_scale) * float(sig)
                except Exception:
                    pass

        if ref_type_s == "delay" and ref_value is not None:
            try:
                dly = pd.to_numeric(df_sel[cols.delay_fs_col], errors="coerce").values.astype(float)
                yv = pd.to_numeric(df_sel[str(prop)], errors="coerce").values.astype(float)
                if not isinstance(ref_value, (list, tuple, np.ndarray)):
                    ref_list = [ref_value]
                else:
                    ref_list = list(ref_value)

                ref_list_num = []
                for rv in ref_list:
                    try:
                        ref_list_num.append(float(rv))
                    except Exception:
                        pass

                if len(ref_list_num) > 0:
                    tol = 1e-9
                    m = np.zeros_like(dly, dtype=bool)
                    for rv in ref_list_num:
                        m |= np.isfinite(dly) & (np.abs(dly - rv) <= tol)

                    yref = yv[np.isfinite(yv) & m]
                    if yref.size >= 2:
                        y0 = float(np.nanmedian(yref))
                        sig = cls._robust_sigma(yref, estimator=estimator, ddof=ddof)
                        return y0, float(sigma_scale) * float(sig)
            except Exception:
                pass

        return cls.baseline_from_negative_delay(
            x=xx,
            y=yy,
            sigma_scale=float(sigma_scale),
            estimator=str(estimator),
            ddof=int(ddof),
        )

    # ----------------------------
    # Matplotlib plumbing
    # ----------------------------
    @staticmethod
    def _collect_errorbar_artists(err_container):
        artists = []
        try:
            data_line = err_container.lines[0]
            caplines = err_container.lines[1]
            barcols = err_container.lines[2]
            if data_line is not None:
                artists.append(data_line)
            if caplines is not None:
                try:
                    artists.extend(list(caplines))
                except Exception:
                    pass
            if barcols is not None:
                try:
                    artists.extend(list(barcols))
                except Exception:
                    pass
        except Exception:
            pass
        return artists

    @staticmethod
    def _legend_handles(leg):
        try:
            hh = getattr(leg, "legend_handles", None)
            if hh is not None and len(hh) > 0:
                return list(hh)
        except Exception:
            pass
        try:
            hh = getattr(leg, "legendHandles", None)
            if hh is not None and len(hh) > 0:
                return list(hh)
        except Exception:
            pass
        try:
            return list(leg.get_lines())
        except Exception:
            return []

    @staticmethod
    def _series_colors(n_series: int, cmap: Optional[str] = None):
        if int(n_series) <= 0:
            return []
        if cmap is None or str(cmap).strip() == "":
            return [None] * int(n_series)

        cm = plt.get_cmap(str(cmap))
        n = int(n_series)
        if n == 1:
            return [cm(0.5)]
        return [cm(v) for v in np.linspace(0.0, 1.0, n)]

    def plot(
        self,
        series_list,
        *,
        title=None,
        xlabel="Delay",
        ylabel="Value",
        legend_title=None,
        as_lines=False,
        show_baseline_sigma=False,
        baseline_mode="errorbar",
        baseline_alpha=0.18,
        cmap: Optional[str] = None,
        show=True,
        save=False,
        save_dir=None,
        save_name=None,
        save_format="png",
        save_dpi=300,
        save_overwrite=True,
        close_after=False,
        legend_outside=True,
    ):
        self._apply_style()

        fig, ax = plt.subplots(figsize=(9.5, 6))
        legend_map = {}

        colors = self._series_colors(len(list(series_list)), cmap=cmap)

        for i, s in enumerate(list(series_list)):
            x = np.asarray(s.get("x", []), float)
            y = np.asarray(s.get("y", []), float)
            lab = str(s.get("label", f"exp{i+1}"))
            color = colors[i] if i < len(colors) else None

            xerr_in = s.get("xerr", None)
            if xerr_in is None:
                xerr = None
            else:
                try:
                    xerr_arr = np.asarray(xerr_in, float)
                    if xerr_arr.shape == ():
                        xerr = np.full_like(x, float(xerr_arr), dtype=float)
                    else:
                        xerr = np.asarray(xerr_arr, float)
                except Exception:
                    xerr = None

            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]

            if xerr is not None:
                if xerr.shape == x.shape or xerr.shape == m.shape:
                    xerr = np.asarray(xerr, float)[m]
                else:
                    try:
                        xerr = np.full_like(x, float(np.asarray(xerr, float).ravel()[0]), dtype=float)
                    except Exception:
                        xerr = None

            if x.size == 0:
                continue

            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

            if xerr is not None:
                xerr = xerr[idx]
                xerr[~np.isfinite(xerr)] = 0.0
                if not np.any(xerr > 0):
                    xerr = None

            baseline_mode_eff = str(baseline_mode).strip().lower()

            if bool(show_baseline_sigma) and baseline_mode_eff == "errorbar":
                sig = s.get("baseline_sig", None)
                try:
                    sigf = float(sig)
                except Exception:
                    sigf = float("nan")

                yerr = None
                if np.isfinite(sigf) and sigf >= 0:
                    yerr = np.full_like(y, sigf, dtype=float)

                if (yerr is not None) or (xerr is not None):
                    fmt = "o-" if bool(as_lines) else "o"
                    err = ax.errorbar(
                        x,
                        y,
                        xerr=xerr,
                        yerr=yerr,
                        fmt=fmt,
                        capsize=3,
                        label=lab,
                        alpha=1,
                        markersize=PlotStyle.marker_size - 2,
                        color=color,
                    )
                    artists = self._collect_errorbar_artists(err)

                    try:
                        data_line, caplines, barcols = err.lines
                    except Exception:
                        data_line, caplines, barcols = None, [], []

                    if yerr is not None and ("baseline_y0" in s) and (data_line is not None):
                        series_color = data_line.get_color()
                        baseline = ax.axhline(
                            s["baseline_y0"],
                            color=series_color,
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.5,
                            label="_nolegend_",
                            zorder=data_line.get_zorder() - 1,
                        )

                        for bc in barcols:
                            try:
                                bc.set_alpha(0.5)
                            except Exception:
                                pass
                        for c in caplines:
                            try:
                                c.set_alpha(0.5)
                            except Exception:
                                pass

                        baseline.set_picker(True)
                        artists.append(baseline)

                    if len(artists) == 0:
                        artists = []
                else:
                    if bool(as_lines):
                        ln = ax.plot(
                            x, y,
                            marker="o",
                            markersize=PlotStyle.marker_size,
                            linestyle="-",
                            label=lab,
                            color=color,
                        )[0]
                    else:
                        ln = ax.plot(
                            x, y,
                            marker="o",
                            markersize=PlotStyle.marker_size,
                            linestyle="None",
                            label=lab,
                            color=color,
                        )[0]
                    artists = [ln]

                s["_mpl_artists"] = artists
                s["_mpl_main_artist"] = artists[0] if len(artists) else None

            else:
                if xerr is not None:
                    fmt = "o-" if bool(as_lines) else "o"
                    err = ax.errorbar(
                        x,
                        y,
                        xerr=xerr,
                        fmt=fmt,
                        capsize=3,
                        label=lab,
                        alpha=1,
                        markersize=PlotStyle.marker_size - 2,
                        color=color,
                    )
                    artists = self._collect_errorbar_artists(err)
                else:
                    if bool(as_lines):
                        ln = ax.plot(
                            x, y,
                            marker="o",
                            markersize=PlotStyle.marker_size,
                            linestyle="-",
                            label=lab,
                            color=color,
                        )[0]
                    else:
                        ln = ax.plot(
                            x, y,
                            marker="o",
                            markersize=PlotStyle.marker_size,
                            linestyle="None",
                            label=lab,
                            color=color,
                        )[0]
                    artists = [ln]

                s["_mpl_artists"] = artists
                s["_mpl_main_artist"] = artists[0] if len(artists) else None

                if bool(show_baseline_sigma) and baseline_mode_eff == "band":
                    y0 = s.get("baseline_y0", None)
                    sig = s.get("baseline_sig", None)
                    try:
                        y0f = float(y0)
                        sigf = float(sig)
                    except Exception:
                        y0f, sigf = None, None

                    if y0f is not None and sigf is not None and np.isfinite(y0f) and np.isfinite(sigf) and sigf >= 0:
                        ax.axhspan(y0f - sigf, y0f + sigf, alpha=float(baseline_alpha), color=color)

        ax.set_title(str(title) if title is not None else "")
        ax.set_xlabel(str(xlabel))
        ax.set_ylabel(str(ylabel))
        ax.grid(alpha=0.25)

        if bool(legend_outside):
            leg = ax.legend(
                title=legend_title,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
            )
            fig.tight_layout()
        else:
            leg = ax.legend(title=legend_title)
            fig.tight_layout()

        if leg is not None:
            label_to_artists = {}
            for s in list(series_list):
                lab = str(s.get("label", ""))
                arts = list(s.get("_mpl_artists", [])) if isinstance(s.get("_mpl_artists", []), list) else []
                if lab != "" and len(arts) > 0:
                    label_to_artists[lab] = arts

            handles = self._legend_handles(leg)
            texts = list(leg.get_texts())

            for txt, hnd in zip(texts, handles):
                lab = str(txt.get_text())
                arts = label_to_artists.get(lab, [])

                try:
                    txt.set_picker(True)
                except Exception:
                    pass
                try:
                    hnd.set_picker(True)
                except Exception:
                    pass

                legend_map[id(txt)] = {"arts": arts, "txt": txt, "hnd": hnd}
                legend_map[id(hnd)] = {"arts": arts, "txt": txt, "hnd": hnd}

            def _set_entry_alpha(txt, hnd, a):
                try:
                    txt.set_alpha(a)
                except Exception:
                    pass
                try:
                    if hasattr(hnd, "set_alpha"):
                        hnd.set_alpha(a)
                except Exception:
                    pass

            def _on_pick(event):
                rec = legend_map.get(id(event.artist), None)
                if not rec:
                    return
                arts = rec.get("arts", [])
                if not arts:
                    return

                try:
                    vis = bool(arts[0].get_visible())
                except Exception:
                    vis = True

                new_vis = not vis
                for a in arts:
                    try:
                        a.set_visible(new_vis)
                    except Exception:
                        pass

                _set_entry_alpha(rec.get("txt", None), rec.get("hnd", None), 1.0 if new_vis else 0.2)

                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

            try:
                fig.canvas.mpl_connect("pick_event", _on_pick)
            except Exception:
                pass

        saved_path = None
        if bool(save):
            if save_dir is None or str(save_dir).strip() == "":
                save_dir = self._default_general_figures_dir()

            os.makedirs(str(save_dir), exist_ok=True)

            fmt = str(save_format).lstrip(".").lower()
            if save_name is None or str(save_name).strip() == "":
                save_name = "time_evolution_multi"

            out_path = os.path.join(str(save_dir), f"{str(save_name)}.{fmt}")

            if (not bool(save_overwrite)) and os.path.exists(out_path):
                raise FileExistsError(out_path)

            fig.savefig(out_path, dpi=int(save_dpi), bbox_inches="tight")
            saved_path = str(out_path)

            if bool(close_after):
                try:
                    plt.close(fig)
                except Exception:
                    pass

        if bool(show):
            try:
                plt.show()
            except Exception:
                pass

        return fig, ax, saved_path


class FitFluenceEvolutionPlotter:
    """
    Thin wrapper around FitTimeEvolutionPlotter for fluence scans.

    Notes:
      - fitting.py currently calls plot(..., fluence_unit=...)
        so we keep that keyword for compatibility.
      - Baseline sigma logic in FitTimeEvolutionPlotter uses x < 0 as baseline region.
        For fluence scans (typically x>0), this means baseline shading/errorbars will
        usually not appear unless you intentionally shift with fluence_offset.
    """

    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = DEFAULT_STYLE if style is None else style

    def plot(
        self,
        df,
        *,
        peak: str,
        y: str,
        fluence_unit: str = "mJ/cm$^2$",
        group_by: str = "azim_range_str",
        groups: Optional[Sequence[Union[str, float, int, Tuple[float, float]]]] = None,
        only_success: bool = True,
        include_reference: bool = True,
        title: Optional[str] = None,
        as_lines: bool = False,
        fluence_offset: float = 0.0,
        figsize: Tuple[float, float] = (7, 6),
        alpha: float = 0.85,
        ms: float = 4.0,
        ref_alpha: float = 0.5,
        ref_linestyle: str = ":",
        # ---- baseline uncertainty
        show_baseline_sigma: bool = False,
        baseline_sigma: float = 1.0,
        baseline_alpha: float = 0.18,
        baseline_mode: str = "band",
        baseline_estimator: str = "std",
        baseline_ddof: int = 1,
        # ---- legend controls
        legend_title: Optional[str] = None,
        legend_loc: str = "upper left",
        legend_outside: bool = True,
        legend_bbox: Tuple[float, float] = (1.02, 1.0),
        # ---- explicit margins
        left: float = 0.20,
        bottom: float = 0.20,
        top: float = 0.8,
        right_inside: float = 0.96,
        right_outside: float = 0.70,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
        # ---- advanced
        x_col: str = "fluence_mJ_cm2",
        x_label: Optional[str] = None,
        x_scale: Optional[float] = None,
    ):
        xlab = x_label if x_label is not None else f"Fluence [{fluence_unit}]"

        core = FitTimeEvolutionPlotter(style=self.style)
        return core.plot(
            df,
            peak=str(peak),
            y=str(y),
            unit=str(fluence_unit),  # only used if x_col == 'delay_fs'
            group_by=str(group_by),
            groups=groups,
            only_success=bool(only_success),
            include_reference=bool(include_reference),
            title=title,
            as_lines=bool(as_lines),
            figsize=tuple(figsize),
            alpha=float(alpha),
            ms=float(ms),
            ref_alpha=float(ref_alpha),
            ref_linestyle=str(ref_linestyle),
            show_baseline_sigma=bool(show_baseline_sigma),
            baseline_sigma=float(baseline_sigma),
            baseline_alpha=float(baseline_alpha),
            baseline_mode=str(baseline_mode),
            baseline_estimator=str(baseline_estimator),
            baseline_ddof=int(baseline_ddof),
            legend_title=legend_title,
            legend_loc=str(legend_loc),
            legend_outside=bool(legend_outside),
            legend_bbox=tuple(legend_bbox),
            left=float(left),
            bottom=float(bottom),
            top=float(top),
            right_inside=float(right_inside),
            right_outside=float(right_outside),
            save=bool(save),
            save_dir=save_dir,
            save_name=save_name,
            save_format=str(save_format),
            save_dpi=int(save_dpi),
            save_overwrite=bool(save_overwrite),
            # key part for fluence scans:
            x_col=str(x_col),
            x_label=str(xlab),
            x_scale=(None if x_scale is None else float(x_scale)),
            x_offset=float(fluence_offset),
        )


class FitFluenceEvolutionMultiPlotter:
    """
    Plot multiple experiments on the same axes (fluence evolution).

    Delegates plotting (and clickable legend behavior) to FitTimeEvolutionMultiPlotter.plot(),
    but provides fluence-specific defaults + shared helpers used by fitting.py.

    Expected series dict keys:
      - x: fluence array
      - y: property array
      - label: legend label (must be unique for clickable legend!)
      - baseline_sig (optional): constant sigma for errorbars if show_baseline_sigma=True
      - baseline_y0 (optional): baseline mean for 'band' mode (rarely used here)
    """

    def __init__(self, style=None, paths: AnalysisPaths | None = None):
        self.style = style
        self.paths = paths

    # ----------------------------
    # Defaults / naming / labels
    # ----------------------------
    def _default_general_figures_dir(self) -> str:
        if self.paths is None:
            raise ValueError("A paths object is required to build default save directories.")
        return str(self.paths.analysis_root / "general_figures")

    @staticmethod
    def legend_title_default() -> str:
        return legend_title_default(scan_type="fluence")

    @staticmethod
    def _format_int_or_float(x) -> str:
        try:
            xf = float(x)
        except Exception:
            return str(x)
        if abs(xf - round(xf)) < 1e-12:
            return str(int(round(xf)))
        return f"{xf:g}"

    @staticmethod
    def _extract_delay_fs(exp: dict) -> str:
        for k in ("delay_fs", "delay_fs_fixed", "delay", "delay_fs_val"):
            if k in exp and exp.get(k, None) is not None:
                try:
                    return str(int(float(exp[k])))
                except Exception:
                    try:
                        return FitFluenceEvolutionMultiPlotter._format_int_or_float(exp[k])
                    except Exception:
                        return str(exp[k])
        return ""

    @classmethod
    def default_label_from_experiment(cls, exp: dict) -> str:
        # If user provided a label, respect it (but ensure uniqueness later).
        lab = str(exp.get("label", "")).strip()
        if lab != "":
            return lab

        # Build a deterministic, informative label that includes delay.
        sn = str(exp.get("sample_name", "")).strip()
        tK = exp.get("temperature_K", None)
        wl = exp.get("excitation_wl_nm", None)
        tw = exp.get("time_window_fs", None)

        parts = []
        if sn:
            parts.append(sn)
        if tK is not None:
            parts.append(f"{cls._format_int_or_float(tK)}")
        if wl is not None:
            parts.append(f"{cls._format_int_or_float(wl)}")
        

        dly = cls._extract_delay_fs(exp)
        if dly != "":
            parts.append(f"{dly}")
        if tw is not None:
            parts.append(f"{cls._format_int_or_float(tw)}")

        out = ", ".join([p for p in parts if str(p).strip() != ""]).strip()

        return out if out != "" else "exp"

    @staticmethod
    def ylabel_for_property(prop: str, peak: str) -> str:
        return FitTimeEvolutionMultiPlotter.ylabel_for_property(prop=str(prop), peak=str(peak))

    @staticmethod
    def title_default(*, peak: str, prop: str, group_by: str = "", group_key: str = "") -> str:
        t = f"hkl=({str(peak)}), {str(prop)}"
        gb = str(group_by).strip()
        gk = str(group_key).strip()
        if gb and gk:
            t += f"\n{gb}={gk}"
        return t

    @staticmethod
    def _sanitize_token(s: str) -> str:
        s = str(s)
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("_", "-", "."):
                out.append(ch)
            else:
                out.append("_")
        tok = "".join(out)
        while "__" in tok:
            tok = tok.replace("__", "_")
        return tok.strip("_")

    @classmethod
    def default_save_name(
        cls,
        *,
        peak: str,
        prop: str,
        group_by: str = "",
        group_key: str = "",
        phi_mode: str = "",
        phi_reduce: str = "",
        n_series: int = 0,
        unit: str = "",
    ) -> str:
        gb = cls._sanitize_token(str(group_by)) if group_by else ""
        gk = cls._sanitize_token(str(group_key)) if group_key else ""
        grp_tok = (gb + "_" + gk).strip("_") if (gb or gk) else "group"
        mode_tok = cls._sanitize_token(str(phi_mode)) if phi_mode else "mode"
        red_tok = cls._sanitize_token(str(phi_reduce)) if phi_reduce else ""
        u = cls._sanitize_token(str(unit)) if unit else ""

        base = f"compare_fluence_peak{cls._sanitize_token(peak)}_{cls._sanitize_token(prop)}_{grp_tok}"
        if mode_tok:
            base += f"_{mode_tok}"
        if red_tok:
            base += f"_{red_tok}"
        base += f"_N{int(n_series)}"
        if u:
            base += f"_{u}"
        return base

    @staticmethod
    def resolve_group_key(df, group_by: str, group):
        """
        Map user-facing group names (e.g. 'Full') into the actual CSV key (e.g. '-90_90').
        """
        if group is None:
            return None

        gb = str(group_by)
        g = group

        candidates = []

        if gb == "azim_range_str":
            if isinstance(g, (list, tuple)) and len(g) == 2:
                try:
                    candidates.append(general_utils.azim_range_str((float(g[0]), float(g[1]))))
                except Exception:
                    pass

            gs = str(g)
            candidates.append(gs)

            if gs.strip().lower() == "full":
                try:
                    candidates.append(general_utils.azim_range_str((-90.0, 90.0)))
                except Exception:
                    pass
                candidates.append("-90_90")
        else:
            candidates.append(str(g))
            if str(g).strip() == "-90_90":
                candidates.append("Full")

        vals = set(df[gb].astype(str).unique().tolist()) if (df is not None and gb in df.columns) else set()
        for c in candidates:
            if str(c) in vals:
                return str(c)

        return str(group)

    @staticmethod
    def _ensure_unique_series_labels(series_list):
        out = []
        seen = {}
        for i, s in enumerate(list(series_list)):
            d = dict(s)
            lab = str(d.get("label", "")).strip()
            if lab == "":
                lab = f"series_{i+1}"
            if lab in seen:
                seen[lab] += 1
                lab_u = f"{lab} ({seen[lab]})"
            else:
                seen[lab] = 1
                lab_u = lab
            d["label"] = lab_u
            out.append(d)
        return out

    # ----------------------------
    # Plot delegate
    # ----------------------------
    def plot(
        self,
        series_list,
        *,
        title=None,
        fluence_unit="mJ/cm$^2$",
        ylabel="Value",
        legend_title=None,
        as_lines=False,
        show_baseline_sigma=False,
        baseline_mode="errorbar",
        show=True,
        save=False,
        save_dir=None,
        save_name=None,
        save_format="png",
        save_dpi=300,
        save_overwrite=True,
        close_after=False,
        legend_outside=True,
    ):
        core = FitTimeEvolutionMultiPlotter(style=self.style)

        xlabel = f"Fluence [{fluence_unit}]"
        if legend_title is None:
            legend_title = self.legend_title_default()

        if save and (save_dir is None or str(save_dir).strip() == ""):
            save_dir = self._default_general_figures_dir()

        series_list_u = self._ensure_unique_series_labels(series_list)

        return core.plot(
            series_list_u,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend_title=legend_title,
            as_lines=as_lines,
            show_baseline_sigma=show_baseline_sigma,
            baseline_mode=baseline_mode,
            show=show,
            save=save,
            save_dir=save_dir,
            save_name=save_name,
            save_format=save_format,
            save_dpi=save_dpi,
            save_overwrite=save_overwrite,
            close_after=close_after,
            legend_outside=legend_outside,
        )


# ============================================================
# Crystallite distribution plots
# ============================================================

class CrystDistributionPlotter:
    """
    Plotting utilities for crystallite / azimuthal distribution outputs.
    """

    def __init__(self, style: PlotStyle = DEFAULT_STYLE):
        self.style = style

    @staticmethod
    def _annotate_sum(ax, value, label="$\\sum$", ha="left", va="top"):
        ax.text(
            0.02, 0.98, f"{label}={value:.4f}",
            transform=ax.transAxes,
            ha=ha, va=va,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    def plot_unfolded_map(
        self,
        data,
        q_array,
        chi_array,
        *,
        phi_shift: float = -90.0,
        clim=None,
        title=None,
        xlim=None,
        q_center=None,
        q_width=None,
        figsize: Tuple[float, float] = (5.6, 5.2),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()

        data = np.asarray(data)
        q_array = np.asarray(q_array)
        chi_array = np.asarray(chi_array)

        fig, ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title, y=1.02, fontsize=self.style.title_fontsize)

        extent = [q_array[0], q_array[-1], chi_array[0] + phi_shift, chi_array[-1] + phi_shift]

        if clim is None:
            ax.imshow(data, extent=extent, origin="lower", aspect="auto")
        else:
            ax.imshow(data, extent=extent, origin="lower", aspect="auto", clim=clim)

        if (q_center is not None) and (q_width is not None):
            q_center = float(q_center)
            q_width = float(q_width)

            q_min = q_center - q_width / 2.0
            q_max = q_center + q_width / 2.0
            q_bg_before_min = q_min - q_width
            q_bg_before_max = q_min
            q_bg_after_min = q_max
            q_bg_after_max = q_max + q_width

            ax.axvline(q_min, color="black", linestyle="--", linewidth=1.5)
            ax.axvline(q_max, color="black", linestyle="--", linewidth=1.5)
            ax.axvline(q_bg_before_min, color="gray", linestyle=":", linewidth=1.5)
            ax.axvline(q_bg_before_max, color="gray", linestyle=":", linewidth=1.5)
            ax.axvline(q_bg_after_min, color="gray", linestyle=":", linewidth=1.5)
            ax.axvline(q_bg_after_max, color="gray", linestyle=":", linewidth=1.5)

        ax.set_xlabel("q [Å$^{-1}$]", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Azimuthal Angle, $\\phi$ [°]", fontsize=self.style.label_fontsize)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_ylim(-95, 95)
        ax.grid(axis="y")
        fig.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("CrystDistributionPlotter.plot_unfolded_map(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "unfolded_map"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax

    def plot_phi_profiles(
        self,
        profiles,
        *,
        phi_range=(-90, 90),
        title=None,
        figsize: Tuple[float, float] = (7.2, 6.0),
        legend_outside: bool = True,
        clickable: bool = True,
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()

        phi = np.asarray(profiles["phi"], float)
        m = (phi >= float(phi_range[0])) & (phi <= float(phi_range[1])) & np.isfinite(phi)

        fig, ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title, y=1.02, fontsize=self.style.title_fontsize)

        l1 = ax.plot(phi[m], np.asarray(profiles["bg_before"], float)[m], "-", linewidth=2, label="BG before (raw)")[0]
        l2 = ax.plot(phi[m], np.asarray(profiles["bg_after"], float)[m], "-", linewidth=2, label="BG after (raw)")[0]
        l3 = ax.plot(phi[m], np.asarray(profiles["bg_avg"], float)[m], "--", linewidth=2, label="BG avg (raw)")[0]
        l4 = ax.plot(phi[m], np.asarray(profiles["peak_raw"], float)[m], 
                     "o", markersize=PlotStyle.marker_size, color="gray", label="Peak (raw)")[0]
        corr_label = f"Corrected (peak - {profiles['bg_mode']})"
        l5 = ax.plot(phi[m], np.asarray(profiles["corrected"], float)[m], 
                     "o", markersize=PlotStyle.marker_size, 
                     color="black", label=corr_label)[0]

        ax.set_xlim(phi_range)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xlabel("Azimuthal Angle, $\\phi$ [°]", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Integrated intensity [a.u.]", fontsize=self.style.label_fontsize)
        ax.grid()

        if legend_outside:
            leg = ax.legend(framealpha=1, loc="upper left", bbox_to_anchor=(1, 1.02), fontsize=self.style.overall_fontsize)
        else:
            leg = ax.legend(framealpha=1, loc="upper right", fontsize=self.style.overall_fontsize)

        if clickable and leg is not None:
            _make_legend_clickable(
                ax,
                legend=leg,
                label_to_artists={
                    "BG before (raw)": [l1],
                    "BG after (raw)": [l2],
                    "BG avg (raw)": [l3],
                    "Peak (raw)": [l4],
                    corr_label: [l5],
                },
            )

        fig.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("CrystDistributionPlotter.plot_phi_profiles(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "phi_profiles"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax

    def plot_fraction_vs_phi(
        self,
        df_frac,
        *,
        phi_range=(-90, 90),
        title=None,
        figsize: Tuple[float, float] = (6.2, 5.2),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()

        yv = df_frac["fraction"].astype(float).values
        s = float(np.nansum(yv))

        fig, ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title, y=1.02, fontsize=self.style.title_fontsize)

        ax.plot(df_frac["phi_center"], yv, "-o", markersize=PlotStyle.marker_size)
        ax.set_xlim(phi_range)
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_xlabel("Azimuthal Angle, $\\phi$ [°]", fontsize=self.style.label_fontsize)
        ax.set_ylabel("Fraction per window", fontsize=self.style.label_fontsize)
        ax.grid()
        self._annotate_sum(ax, s, label="$\\sum$ points")

        fig.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("CrystDistributionPlotter.plot_fraction_vs_phi(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "fraction_vs_phi"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax

    def plot_folded_abs_phi(
        self,
        df_fold,
        *,
        phi_range=(0, 90),
        title=None,
        kind: str = "mass",  # "mass" | "per_side"
        figsize: Tuple[float, float] = (6.2, 5.2),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        self.style.apply()

        if kind not in ("mass", "per_side"):
            raise ValueError("kind must be 'mass' or 'per_side'")

        ycol = "fraction_mass" if kind == "mass" else "fraction_per_side"
        ylabel = "Folded fraction (sum over ±φ)" if kind == "mass" else "Folded fraction per-side (mean over ±φ)"

        yv = df_fold[ycol].astype(float).values
        s = float(np.nansum(yv))

        fig, ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title, y=1.02, fontsize=self.style.title_fontsize)

        ax.plot(df_fold["abs_phi"], yv, "-o", markersize=PlotStyle.marker_size)
        ax.set_xlim(phi_range)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.set_xlabel("$|\\phi|$ [°]", fontsize=self.style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.style.label_fontsize)
        ax.grid()
        self._annotate_sum(ax, s, label="$\\sum$ points")

        fig.tight_layout()

        if save:
            if save_dir is None:
                raise ValueError("CrystDistributionPlotter.plot_folded_abs_phi(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or f"folded_abs_phi_{kind}"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, ax


# ============================================================
# Differential plots
# ============================================================

@dataclass
class DifferentialPlotStyle:
    grid: bool = True
    show_zero_lines: bool = True


class DifferentialTimeTracePlotter:
    """
    Two-subplot plotter (clickable legend):
      - top: integral(ΔI)
      - bottom: integral(|ΔI|)
    """

    def __init__(self, style: Optional[PlotStyle] = None, local_style: Optional[DifferentialPlotStyle] = None):
        self.style = DEFAULT_STYLE if style is None else style
        self.local_style = DifferentialPlotStyle() if local_style is None else local_style

    def plot(
        self,
        df: pd.DataFrame,
        *,
        title: Optional[str] = None,
        unit: str = "ps",
        delay_offset: float = 0.0,
        group_by: str = "region",
        groups: Optional[Sequence[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        # ---- error bars
        show_errorbars: bool = False,
        errorbars_from_group: str = "background",
        errorbars_for_groups: Optional[Sequence[str]] = ("peak",),
        errorbar_scale: float = 1.0,
        xlim: Optional[Tuple[float, float]] = None,
        ylim_signed: Optional[Tuple[float, float]] = None,
        ylim_abs: Optional[Tuple[float, float]] = None,
        legend_outside: bool = True,
        legend_loc: str = "upper left",
        legend_bbox: Tuple[float, float] = (1.02, 1.0),
        figsize: Tuple[float, float] = (6.0, 7.0),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ):
        self.style.apply()

        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe passed to DifferentialTimeTracePlotter.plot().")

        unit = str(unit).strip().lower()
        if unit not in ("fs", "ps"):
            raise ValueError("unit must be 'fs' or 'ps'")

        xcol = "delay_fs" if unit == "fs" else "delay_ps"
        xscale_from_tw = 1.0 if unit == "fs" else 1e-3

        for c in (xcol, "int_delta", "int_abs_delta", group_by, "delay_fs"):
            if c not in df.columns:
                raise KeyError(f"Expected column '{c}' in df.")

        d = df.copy()
        d = d[np.isfinite(d[xcol].astype(float).values)].copy()
        d = d.sort_values(xcol)

        if groups is None:
            groups = list(pd.unique(d[group_by].astype(str)))

        title_fs = float(self.style.title_fontsize)
        label_fs = float(self.style.label_fontsize)

        s_area = float(self.style.marker_size)
        msize_pts = _markersize_from_scatter_s(s_area)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if title is not None:
            ax0.set_title(str(title), fontsize=title_fs)

        cmap_default = {"peak": "blue", "background": "gray"}
        colors = dict(cmap_default if colors is None else colors)

        bg_err_signed: Dict[int, float] = {}
        bg_err_abs: Dict[int, float] = {}
        if show_errorbars:
            bg_name = str(errorbars_from_group)
            m_bg = d[group_by].astype(str) == bg_name
            if np.any(m_bg):
                dd_fs = d.loc[m_bg, "delay_fs"].astype(float).values
                e_signed = d.loc[m_bg, "int_delta"].astype(float).values
                e_abs = d.loc[m_bg, "int_abs_delta"].astype(float).values

                ok = np.isfinite(dd_fs) & np.isfinite(e_signed) & np.isfinite(e_abs)
                dd_fs = dd_fs[ok].astype(int)
                e_signed = np.abs(e_signed[ok]) * float(errorbar_scale)
                e_abs = np.abs(e_abs[ok]) * float(errorbar_scale)

                for i in range(dd_fs.size):
                    bg_err_signed[int(dd_fs[i])] = float(e_signed[i])
                    bg_err_abs[int(dd_fs[i])] = float(e_abs[i])

        label_to_artists: Dict[str, List[object]] = {}

        err_groups = None
        if errorbars_for_groups is not None:
            err_groups = {str(x) for x in list(errorbars_for_groups)}

        for g in groups:
            g_str = str(g)
            m = d[group_by].astype(str) == g_str
            if not np.any(m):
                continue

            x = d.loc[m, xcol].astype(float).values + float(delay_offset)
            y_signed = d.loc[m, "int_delta"].astype(float).values
            y_abs = d.loc[m, "int_abs_delta"].astype(float).values
            d_fs = d.loc[m, "delay_fs"].astype(float).values

            if "time_window_fs" in d.columns:
                tw_fs = pd.to_numeric(d.loc[m, "time_window_fs"], errors="coerce").values.astype(float)
            else:
                tw_fs = np.full_like(x, np.nan, dtype=float)

            ok = np.isfinite(x) & np.isfinite(y_signed) & np.isfinite(y_abs) & np.isfinite(d_fs)
            x = x[ok]
            y_signed = y_signed[ok]
            y_abs = y_abs[ok]
            d_fs = d_fs[ok].astype(int)
            tw_fs = tw_fs[ok]

            order = np.argsort(x)
            x = x[order]
            y_signed = y_signed[order]
            y_abs = y_abs[order]
            d_fs = d_fs[order]
            tw_fs = tw_fs[order]

            col = colors.get(g_str, None)

            xerr = None
            if tw_fs.size > 0:
                xerr_arr = 0.5 * tw_fs * float(xscale_from_tw)
                xerr_arr = np.asarray(xerr_arr, float)
                xerr_arr[~np.isfinite(xerr_arr)] = 0.0
                if np.any(xerr_arr > 0.0):
                    xerr = xerr_arr

            use_err = bool(show_errorbars) and (err_groups is not None) and (g_str in err_groups) and (len(bg_err_signed) > 0)

            artists_this: List[object] = []

            if use_err:
                yerr0 = np.array([bg_err_signed.get(int(dd), np.nan) for dd in d_fs], float)
                yerr1 = np.array([bg_err_abs.get(int(dd), np.nan) for dd in d_fs], float)

                yerr0[~np.isfinite(yerr0)] = 0.0
                yerr1[~np.isfinite(yerr1)] = 0.0

                eb0 = ax0.errorbar(
                    x, y_signed,
                    xerr=xerr,
                    yerr=yerr0,
                    fmt="o",
                    linestyle="none",
                    color=col,
                    markersize=msize_pts,
                    capsize=0,
                    elinewidth=1.2,
                    alpha=0.4,
                    label=g_str,
                )
                eb1 = ax1.errorbar(
                    x, y_abs,
                    xerr=xerr,
                    yerr=yerr1,
                    fmt="o",
                    linestyle="none",
                    color=col,
                    markersize=msize_pts,
                    capsize=0,
                    elinewidth=1.2,
                    alpha=0.4,
                    label=g_str,
                )

                artists_this.extend(_flatten_errorbar_container(eb0))
                artists_this.extend(_flatten_errorbar_container(eb1))
            else:
                if xerr is not None:
                    eb0 = ax0.errorbar(
                        x, y_signed,
                        xerr=xerr,
                        fmt="o",
                        linestyle="none",
                        color=col,
                        markersize=msize_pts,
                        capsize=0,
                        elinewidth=1.2,
                        alpha=0.4,
                        label=g_str,
                    )
                    eb1 = ax1.errorbar(
                        x, y_abs,
                        xerr=xerr,
                        fmt="o",
                        linestyle="none",
                        color=col,
                        markersize=msize_pts,
                        capsize=0,
                        elinewidth=1.2,
                        alpha=0.4,
                        label=g_str,
                    )
                    artists_this.extend(_flatten_errorbar_container(eb0))
                    artists_this.extend(_flatten_errorbar_container(eb1))
                else:
                    sc0 = ax0.scatter(x, y_signed, s=s_area, color=col, label=g_str)
                    sc1 = ax1.scatter(x, y_abs, s=s_area, color=col, label=g_str)
                    artists_this.extend([sc0, sc1])

            if artists_this:
                label_to_artists[g_str] = artists_this

        ax0.set_ylabel(r"$\int$∆Idq [a.u.]", fontsize=label_fs)
        ax1.set_ylabel(r"$\int$|∆I|dq [a.u.]", fontsize=label_fs)
        ax1.set_xlabel(f"Delay [{unit}]", fontsize=label_fs)

        ax0.tick_params(axis="both", which="both")
        ax1.tick_params(axis="both", which="both")

        if self.local_style.show_zero_lines:
            ax0.axhline(0.0, lw=1.5, ls="--", alpha=0.4)
            ax1.axhline(0.0, lw=1.5, ls="--", alpha=0.4)

        if self.local_style.grid:
            ax0.grid()
            ax1.grid()

        if xlim is not None:
            ax1.set_xlim(tuple(xlim))
        if ylim_signed is not None:
            ax0.set_ylim(tuple(ylim_signed))
        if ylim_abs is not None:
            ax1.set_ylim(tuple(ylim_abs))

        if legend_outside:
            leg = ax0.legend(
                loc=str(legend_loc),
                bbox_to_anchor=tuple(legend_bbox),
                borderaxespad=0.0,
                framealpha=1.0,
            )
            fig.tight_layout(rect=[0, 0, 1, 1])
        else:
            leg = ax0.legend(loc=str(legend_loc), framealpha=1.0)
            fig.tight_layout(rect=[0, 0, 1.0, 1])

        _make_legend_clickable(ax0, legend=leg, label_to_artists=label_to_artists)

        if save:
            if save_dir is None:
                raise ValueError("DifferentialTimeTracePlotter.plot(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "differential_time_trace"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, (ax0, ax1)


class DifferentialFluenceTracePlotter:
    """
    Two-subplot plotter (clickable legend), but for FLUENCE scans:
      - top: integral(ΔI)
      - bottom: integral(|ΔI|)

    Expected df columns:
      fluence_mJ_cm2, region, int_delta, int_abs_delta
    """

    def __init__(self, style: Optional[PlotStyle] = None, local_style: Optional[DifferentialPlotStyle] = None):
        self.style = DEFAULT_STYLE if style is None else style
        self.local_style = DifferentialPlotStyle() if local_style is None else local_style

    @staticmethod
    def _key_fluence(x: float, ndp: int = 6) -> float:
        # robust dict-key for float fluence values
        try:
            return float(np.round(float(x), int(ndp)))
        except Exception:
            return float("nan")

    def plot(
        self,
        df: pd.DataFrame,
        *,
        title: Optional[str] = None,
        fluence_unit: str = "mJ/cm$^2$",
        fluence_offset: float = 0.0,
        group_by: str = "region",
        groups: Optional[Sequence[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        # ---- error bars
        show_errorbars: bool = False,
        errorbars_from_group: str = "background",
        errorbars_for_groups: Optional[Sequence[str]] = ("peak",),
        errorbar_scale: float = 1.0,
        xlim: Optional[Tuple[float, float]] = None,
        ylim_signed: Optional[Tuple[float, float]] = None,
        ylim_abs: Optional[Tuple[float, float]] = None,
        legend_outside: bool = True,
        legend_loc: str = "upper left",
        legend_bbox: Tuple[float, float] = (1.02, 1.0),
        figsize: Tuple[float, float] = (6.0, 7.0),
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ):
        self.style.apply()

        if df is None or len(df) == 0:
            raise ValueError("Empty dataframe passed to DifferentialFluenceTracePlotter.plot().")

        for c in ("fluence_mJ_cm2", "int_delta", "int_abs_delta", group_by):
            if c not in df.columns:
                raise KeyError(f"Expected column '{c}' in df.")

        d = df.copy()
        d = d[np.isfinite(d["fluence_mJ_cm2"].astype(float).values)].copy()
        d = d.sort_values("fluence_mJ_cm2")

        if groups is None:
            groups = list(pd.unique(d[group_by].astype(str)))

        title_fs = PlotStyle.title_fontsize
        label_fs = PlotStyle.label_fontsize

        s_area = PlotStyle.overall_fontsize
        msize_pts = _markersize_from_scatter_s(s_area)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if title is not None:
            #fig.suptitle(str(title), fontsize=title_fs)
            ax0.set_title(str(title), fontsize=title_fs)


        cmap_default = {"peak": "blue", "background": "gray"}
        colors = dict(cmap_default if colors is None else colors)

        # ---- build per-fluence errorbars from background group
        bg_err_signed: Dict[float, float] = {}
        bg_err_abs: Dict[float, float] = {}
        if show_errorbars:
            bg_name = str(errorbars_from_group)
            m_bg = d[group_by].astype(str) == bg_name
            if np.any(m_bg):
                fvals = d.loc[m_bg, "fluence_mJ_cm2"].astype(float).values
                e_signed = d.loc[m_bg, "int_delta"].astype(float).values
                e_abs = d.loc[m_bg, "int_abs_delta"].astype(float).values

                ok = np.isfinite(fvals) & np.isfinite(e_signed) & np.isfinite(e_abs)
                fvals = fvals[ok]
                e_signed = np.abs(e_signed[ok]) * float(errorbar_scale)
                e_abs = np.abs(e_abs[ok]) * float(errorbar_scale)

                for i in range(fvals.size):
                    k = self._key_fluence(fvals[i])
                    if np.isfinite(k):
                        bg_err_signed[k] = float(e_signed[i])
                        bg_err_abs[k] = float(e_abs[i])

        label_to_artists: Dict[str, List[object]] = {}

        err_groups = None
        if errorbars_for_groups is not None:
            err_groups = {str(x) for x in list(errorbars_for_groups)}

        for g in groups:
            g_str = str(g)
            m = d[group_by].astype(str) == g_str
            if not np.any(m):
                continue

            x = d.loc[m, "fluence_mJ_cm2"].astype(float).values + float(fluence_offset)
            y_signed = d.loc[m, "int_delta"].astype(float).values
            y_abs = d.loc[m, "int_abs_delta"].astype(float).values
            f_raw = d.loc[m, "fluence_mJ_cm2"].astype(float).values

            ok = np.isfinite(x) & np.isfinite(y_signed) & np.isfinite(y_abs) & np.isfinite(f_raw)
            x = x[ok]
            y_signed = y_signed[ok]
            y_abs = y_abs[ok]
            f_raw = f_raw[ok]

            order = np.argsort(x)
            x = x[order]
            y_signed = y_signed[order]
            y_abs = y_abs[order]
            f_raw = f_raw[order]

            col = colors.get(g_str, None)

            use_err = bool(show_errorbars) and (err_groups is not None) and (g_str in err_groups) and (len(bg_err_signed) > 0)

            artists_this: List[object] = []

            if use_err:
                keys = np.array([self._key_fluence(ff) for ff in f_raw], float)
                yerr0 = np.array([bg_err_signed.get(float(k), np.nan) for k in keys], float)
                yerr1 = np.array([bg_err_abs.get(float(k), np.nan) for k in keys], float)

                yerr0[~np.isfinite(yerr0)] = 0.0
                yerr1[~np.isfinite(yerr1)] = 0.0

                eb0 = ax0.errorbar(
                    x, y_signed,
                    yerr=yerr0,
                    fmt="o",
                    linestyle="none",
                    color=col,
                    markersize=msize_pts,
                    capsize=0,
                    elinewidth=1.2,
                    alpha=0.4,
                    label=g_str,
                )
                eb1 = ax1.errorbar(
                    x, y_abs,
                    yerr=yerr1,
                    fmt="o",
                    linestyle="none",
                    color=col,
                    markersize=msize_pts,
                    capsize=0,
                    elinewidth=1.2,
                    alpha=0.4,
                    label=g_str,
                )

                artists_this.extend(_flatten_errorbar_container(eb0))
                artists_this.extend(_flatten_errorbar_container(eb1))
            else:
                sc0 = ax0.scatter(x, y_signed, s=s_area, color=col, label=g_str)
                sc1 = ax1.scatter(x, y_abs, s=s_area, color=col, label=g_str)
                artists_this.extend([sc0, sc1])

            if artists_this:
                label_to_artists[g_str] = artists_this

        ax0.set_ylabel(r"$\int$∆Idq [a.u.]", fontsize=label_fs)
        ax1.set_ylabel(r"$\int$|∆I|dq [a.u.]", fontsize=label_fs)
        ax1.set_xlabel(f"Fluence [{fluence_unit}]", fontsize=label_fs)

        ax0.tick_params(axis="both", which="both")
        ax1.tick_params(axis="both", which="both")

        if self.local_style.show_zero_lines:
            ax0.axhline(0.0, lw=1.5, ls="--", alpha=0.4)
            ax1.axhline(0.0, lw=1.5, ls="--", alpha=0.4)

        if self.local_style.grid:
            ax0.grid()
            ax1.grid()

        if xlim is not None:
            ax1.set_xlim(tuple(xlim))
        if ylim_signed is not None:
            ax0.set_ylim(tuple(ylim_signed))
        if ylim_abs is not None:
            ax1.set_ylim(tuple(ylim_abs))

        if legend_outside:
            leg = ax0.legend(
                loc=str(legend_loc),
                bbox_to_anchor=tuple(legend_bbox),
                borderaxespad=0.0,
                framealpha=1.0,
            )
            fig.tight_layout(rect=[0, 0, 1, 1])
        else:
            leg = ax0.legend(loc=str(legend_loc), framealpha=1.0)
            fig.tight_layout(rect=[0, 0, 1.0, 1])

        _make_legend_clickable(ax0, legend=leg, label_to_artists=label_to_artists)

        if save:
            if save_dir is None:
                raise ValueError("DifferentialFluenceTracePlotter.plot(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "differential_fluence_trace"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, (ax0, ax1)


class DifferentialFFTPlotter:
    """
    Two-subplot FFT plotter (clickable):
      - top: time-domain raw (black), baseline (red), detrended (blue)
      - bottom: |FFT| main (blue) + optional background (gray)
    """

    def __init__(self, style: Optional[PlotStyle] = None, local_style: Optional[DifferentialPlotStyle] = None):
        self.style = DEFAULT_STYLE if style is None else style
        self.local_style = DifferentialPlotStyle() if local_style is None else local_style

    def plot(
        self,
        fft_main: dict,
        *,
        title: Optional[str] = None,
        freq_unit: str = "cm^-1",
        xlim_freq: Optional[Tuple[float, float]] = None,
        ylim_freq: Optional[Tuple[float, float]] = None,
        ylim_time: Optional[Tuple[float, float]] = None,
        delay_offset: float = 0.0,
        show_baseline: bool = True,
        fft_bg: Optional[dict] = None,
        label_main: str = "FFT\n(signal)",
        label_bg: str = "FFT\n(background)",
        # ---- saving
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        save_name: Optional[str] = None,
        save_format: str = "png",
        save_dpi: int = 400,
        save_overwrite: bool = False,
    ):
        self.style.apply()

        title_fs = PlotStyle.title_fontsize
        label_fs = PlotStyle.label_fontsize
        s_area = PlotStyle.overall_fontsize

        t = np.asarray(fft_main.get("t_ps", []), float) + delay_offset
        y_raw = np.asarray(fft_main.get("y_raw", []), float)
        baseline = np.asarray(fft_main.get("baseline", []), float)
        y_detr = np.asarray(fft_main.get("y_detrended", []), float)

        freqs_pos = np.asarray(fft_main.get("freqs_pos", []), float)
        fft_pos = np.asarray(fft_main.get("fft_pos", []), complex)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.5, 7.0))

        if title is not None:
            #fig.suptitle(str(title), fontsize=title_fs)
            ax0.set_title(str(title), fontsize=title_fs)

        a_raw = ax0.scatter(t, y_raw, s=s_area, color="black", label="Raw")
        a_detr = ax0.scatter(t, y_detr, s=s_area, color="blue", label="Detrended")

        a_base = None
        if show_baseline and baseline.size == y_raw.size:
            a_base = ax0.plot(t, baseline, "--", lw=2.0, color="red", label="Poly baseline")[0]

        ax0.set_xlabel("Delay [ps]", fontsize=label_fs)
        ax0.set_ylabel("Signal [a.u.]", fontsize=label_fs)
        ax0.tick_params(axis="both", which="both")
        if ylim_time is not None:
            ax0.set_ylim(tuple(ylim_time))
        ax0.grid()

        label_to_artists_top: Dict[str, List[object]] = {"Raw": [a_raw], "Detrended": [a_detr]}
        if a_base is not None:
            label_to_artists_top["Poly baseline"] = [a_base]

        leg0 = ax0.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            framealpha=1.0,
        )
        _make_legend_clickable(ax0, legend=leg0, label_to_artists=label_to_artists_top)

        amp = np.abs(fft_pos)
        l_main = ax1.plot(freqs_pos, amp, "-", lw=2.0, color="blue", label=label_main)[0]
        label_to_artists_bot: Dict[str, List[object]] = {label_main: [l_main]}

        if fft_bg is not None:
            f2 = np.asarray(fft_bg.get("freqs_pos", []), float)
            z2 = np.asarray(fft_bg.get("fft_pos", []), complex)
            if f2.size and z2.size:
                l_bg = ax1.plot(f2, np.abs(z2), "-", lw=2.0, color="gray", label=label_bg)[0]
                label_to_artists_bot[label_bg] = [l_bg]

        freq_unit_disp = "cm$^{-1}$" if freq_unit == "cm^-1" else str(freq_unit)

        ax1.set_xlabel(f"Frequency [{freq_unit_disp}]", fontsize=label_fs)
        ax1.set_ylabel("|FFT|", fontsize=label_fs)
        ax1.tick_params(axis="both", which="both")
        if xlim_freq is not None:
            ax1.set_xlim(tuple(xlim_freq))
        if ylim_freq is not None:
            ax1.set_ylim(tuple(ylim_freq))
        ax1.grid()

        leg1 = ax1.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=PlotStyle.label_fontsize - 2,
            framealpha=1.0,
        )
        _make_legend_clickable(ax1, legend=leg1, label_to_artists=label_to_artists_bot)

        fig.tight_layout(rect=[0, 0, 1, 1])

        if save:
            if save_dir is None:
                raise ValueError("DifferentialFFTPlotter.plot(save=True) requires save_dir=...")
            save_figure(
                fig,
                save_dir=save_dir,
                save_name=(save_name or title or "differential_fft"),
                fmt=save_format,
                dpi=save_dpi,
                overwrite=save_overwrite,
            )

        plt.show()
        return fig, (ax0, ax1)


class DifferentialTimeTraceMultiPlotter:
    """
    Multi-experiment differential time-trace plotter.

    Expects `series_list` entries like:
      dict(
        experiment=exp_dict,
        delay_offset_ps=float,
        time_ps=np.ndarray,
        int_delta=np.ndarray,
        int_abs_delta=np.ndarray,
        err_delta=np.ndarray,
        err_abs_delta=np.ndarray,
        label=str,
      )
    """

    DEFAULT_SAVE_DIR = None

    def __init__(self, style=None, paths=None, default_save_dir=None):
        self.style = style
        self.paths = paths
        self.default_save_dir = None if default_save_dir is None else Path(default_save_dir)

    @classmethod
    def set_default_save_dir(cls, path):
        cls.DEFAULT_SAVE_DIR = None if path is None else Path(path)

    def _resolve_default_save_dir(self) -> Path:
        if self.default_save_dir is not None:
            return Path(self.default_save_dir)

        if self.DEFAULT_SAVE_DIR is not None:
            return Path(self.DEFAULT_SAVE_DIR)

        if self.paths is not None:
            return Path(self.paths.analysis_root) / "general_figures"

        raise ValueError(
            "No default save directory configured. "
            "Provide save_dir=..., default_save_dir=..., set "
            "DifferentialTimeTraceMultiPlotter.DEFAULT_SAVE_DIR, or pass paths=..."
        )

    @staticmethod
    def _apply_single_experiment_aesthetics(ax):
        g = globals()
        for fname in (
            "_apply_default_axes_style",
            "_format_axes_default",
            "_format_axes",
            "_apply_axes_style",
            "_apply_grid_style",
        ):
            f = g.get(fname, None)
            if callable(f):
                try:
                    f(ax)
                    return
                except Exception:
                    pass
        ax.grid()

    @staticmethod
    def _artists_from_errorbar_container(err_container):
        arts = []
        if err_container is None:
            return arts

        lines = getattr(err_container, "lines", None)
        if lines is not None:
            try:
                for item in lines:
                    if item is None:
                        continue
                    if isinstance(item, (list, tuple)):
                        arts.extend([x for x in item if x is not None])
                    else:
                        arts.append(item)
            except Exception:
                pass

        for attr in ("caplines", "barlinecols"):
            obj = getattr(err_container, attr, None)
            if obj is None:
                continue
            try:
                for x in obj:
                    if x is not None:
                        arts.append(x)
            except Exception:
                pass

        if not arts:
            try:
                for x in err_container:
                    if x is None:
                        continue
                    if isinstance(x, (list, tuple)):
                        arts.extend([y for y in x if y is not None])
                    else:
                        arts.append(x)
            except Exception:
                pass

        return arts

    @staticmethod
    def _get_color_cycle():
        try:
            prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
            if prop_cycle is not None:
                colors = prop_cycle.by_key().get("color", [])
                if colors:
                    return list(colors)
        except Exception:
            pass
        return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    @staticmethod
    def legend_title_default() -> str:
        return globals()["legend_title_default"](scan_type="delay")

    @staticmethod
    def _ensure_legend_clickable(fig, ax, legend, labels, artists_by_label):
        used_external = False
        g = globals()
        f = g.get("_make_legend_clickable", None)

        if callable(f):
            try:
                sig = inspect.signature(f)
                params = sig.parameters

                artists_by_label = dict(artists_by_label) if artists_by_label is not None else {}
                legend_handles = list(getattr(legend, "legendHandles", []))
                if not legend_handles:
                    legend_handles = list(legend.get_lines())

                artists_by_legend = {}
                for h, lbl in zip(legend_handles, labels):
                    arts = artists_by_label.get(lbl, [])
                    artists_by_legend[h] = arts

                candidates = {
                    "fig": fig,
                    "ax": ax,
                    "legend": legend,
                    "artists_by_label": artists_by_label,
                    "label_to_artists": artists_by_label,
                    "artists_by_legend": artists_by_legend,
                    "handle_to_artists": artists_by_legend,
                }

                kwargs = {}
                for name, param in params.items():
                    if name in candidates and candidates[name] is not None:
                        kwargs[name] = candidates[name]

                required = [
                    n
                    for n, p in params.items()
                    if p.default is inspect._empty
                    and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                ]

                if all(r in kwargs for r in required):
                    f(**kwargs)
                    used_external = True
            except Exception:
                used_external = False

        if used_external:
            return

        legend_handles = list(getattr(legend, "legendHandles", []))
        if not legend_handles:
            legend_handles = list(legend.get_lines())

        line_to_artists = {}
        for handle, label in zip(legend_handles, labels):
            arts = artists_by_label.get(label, [])
            line_to_artists[handle] = arts
            try:
                handle.set_picker(True)
                handle.set_pickradius(5)
            except Exception:
                try:
                    handle.set_picker(True)
                except Exception:
                    pass

        def on_pick(event):
            legline = event.artist
            if legline not in line_to_artists:
                return
            arts = line_to_artists[legline]
            if not arts:
                return

            try:
                vis = not arts[0].get_visible()
            except Exception:
                vis = True

            for art in arts:
                try:
                    art.set_visible(vis)
                except Exception:
                    pass

            try:
                legline.set_alpha(0.4 if vis else 0.2)
            except Exception:
                pass

            try:
                fig.canvas.draw_idle()
            except Exception:
                fig.canvas.draw()

        try:
            fig.canvas.mpl_connect("pick_event", on_pick)
        except Exception:
            pass

    def define_fig_name_auto(self, series_list):
        name = "Diff_Time_Analysis_"

        for i in range(len(series_list)):
            exp = series_list[i]["experiment"]

            s_name = exp["sample_name"]
            T_K = exp["temperature_K"]
            ex_wl_nm = exp["excitation_wl_nm"]
            fl_mJ_cm2 = exp["fluence_mJ_cm2"]
            tw_fs = exp["time_window_fs"]
            label = str(exp.get("label", "")).strip()

            if label != "":
                label = f"_label_{label}_"

            dummy = "__" if i < (len(series_list) - 1) else ""

            name += (
                f"Exp{i+1}_{s_name}_{T_K}K_ex_wl_{ex_wl_nm}nm_"
                f"flu_{fl_mJ_cm2}mJcm2_tw_{tw_fs}fs{label}{dummy}"
            )

        return name

    def plot(
        self,
        series_list,
        *,
        unit="ps",
        as_lines=False,
        show_errorbars=True,
        errorbar_scale=1.0,
        title=None,
        legend_title=None,
        show=True,
        save=False,
        save_dir=None,
        save_name=None,
        save_format="png",
        save_dpi=300,
        save_overwrite=True,
        legend_outside=True,
    ):
        if legend_title is None:
            legend_title = self.legend_title_default()

        u = str(unit).strip().lower()
        if u not in ("fs", "ps"):
            raise ValueError("unit must be 'fs' or 'ps'.")

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 7.0))

        self._apply_single_experiment_aesthetics(ax_top)
        self._apply_single_experiment_aesthetics(ax_bot)

        colors = self._get_color_cycle()

        artists_by_label = {}
        legend_handles = []
        legend_labels = []

        xlabel = "Delay [ps]" if u == "ps" else "Delay [fs]"

        for i, s in enumerate(list(series_list)):
            label = str(s.get("label", f"series_{i+1}")).strip()
            delay_offset_ps = float(s.get("delay_offset_ps", 0.0))

            t_ps = np.asarray(s["time_ps"], float)
            y_delta = np.asarray(s["int_delta"], float)
            y_abs = np.asarray(s["int_abs_delta"], float)

            err_delta = np.asarray(
                s.get("err_delta", np.zeros_like(y_delta)), float
            ) * float(errorbar_scale)
            err_abs = np.asarray(
                s.get("err_abs_delta", np.zeros_like(y_abs)), float
            ) * float(errorbar_scale)

            if u == "ps":
                x = t_ps + delay_offset_ps
            else:
                x = (t_ps + delay_offset_ps) * 1e3

            exp = s.get("experiment", {})
            try:
                tw_fs = float(exp.get("time_window_fs", np.nan))
            except Exception:
                tw_fs = np.nan

            if np.isfinite(tw_fs) and (tw_fs > 0):
                half_tw = 0.5 * tw_fs
                if u == "ps":
                    half_tw = half_tw * 1e-3
                xerr = np.full_like(x, float(half_tw), dtype=float)
            else:
                xerr = None

            color = colors[i % len(colors)]
            marker = "o"
            linestyle = "-" if bool(as_lines) else "none"

            arts = []

            if bool(show_errorbars):
                eb_top = ax_top.errorbar(
                    x,
                    y_delta,
                    xerr=xerr,
                    yerr=err_delta,
                    fmt=marker,
                    markersize=5,
                    linestyle=linestyle,
                    color=color,
                    capsize=2,
                    label=label,
                )
                try:
                    data_line, caplines, barlinecols = eb_top.lines
                    for c in caplines:
                        c.set_alpha(0.4)
                    for bc in barlinecols:
                        bc.set_alpha(0.4)
                except Exception:
                    pass

                eb_bot = ax_bot.errorbar(
                    x,
                    y_abs,
                    xerr=xerr,
                    yerr=err_abs,
                    fmt=marker,
                    markersize=5,
                    linestyle=linestyle,
                    color=color,
                    capsize=2,
                    label=None,
                )
                try:
                    data_line, caplines, barlinecols = eb_bot.lines
                    for c in caplines:
                        c.set_alpha(0.4)
                    for bc in barlinecols:
                        bc.set_alpha(0.4)
                except Exception:
                    pass

                arts.extend(self._artists_from_errorbar_container(eb_top))
                arts.extend(self._artists_from_errorbar_container(eb_bot))

                handle = None
                try:
                    handle = eb_top.lines[0]
                except Exception:
                    try:
                        handle = eb_top[0]
                    except Exception:
                        handle = arts[0] if arts else None
            else:
                ln_top = ax_top.plot(
                    x,
                    y_delta,
                    marker=marker,
                    markersize=5,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                )[0]
                ln_bot = ax_bot.plot(
                    x,
                    y_abs,
                    marker=marker,
                    markersize=5,
                    linestyle=linestyle,
                    color=color,
                    label=None,
                )[0]
                arts.extend([ln_top, ln_bot])
                handle = ln_top

            if arts:
                artists_by_label[label] = arts
            if handle is not None:
                legend_handles.append(handle)
                legend_labels.append(label)

        ax_top.set_ylabel(r"$\int$∆Idq [a.u.]")
        ax_bot.set_ylabel(r"$\int$|∆I|dq [a.u.]")
        ax_bot.set_xlabel(xlabel)

        if title is not None:
            ax_top.set_title(title, y=1.01)

        legend = None
        if legend_handles:
            if bool(legend_outside):
                legend = ax_top.legend(
                    legend_handles,
                    legend_labels,
                    title=legend_title,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                )
            else:
                legend = ax_top.legend(
                    legend_handles,
                    legend_labels,
                    title=legend_title,
                )

            if legend is not None:
                self._ensure_legend_clickable(
                    fig, ax_top, legend, legend_labels, artists_by_label
                )

        fig.tight_layout()

        saved_path = None
        if bool(save):
            if save_dir is None:
                save_dir = self._resolve_default_save_dir()
            else:
                save_dir = Path(save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)

            if save_name is None or str(save_name).strip() == "":
                save_name = self.define_fig_name_auto(series_list)

            fmt = str(save_format).lstrip(".").lower()
            saved_path = save_dir / f"{save_name}.{fmt}"

            if (not bool(save_overwrite)) and saved_path.exists():
                raise FileExistsError(str(saved_path))

            fig.savefig(str(saved_path), dpi=int(save_dpi), bbox_inches="tight")

        if bool(show):
            plt.show()
        else:
            plt.close(fig)

        return fig, (ax_top, ax_bot), saved_path

    
class DifferentialFFTMultiPlotter:
    """
    Multi-experiment differential FFT plotter.

    Expects `series_list` entries like:
      dict(
        experiment=exp_dict,
        delay_offset_ps=float,
        time_ps=np.ndarray,
        signal_peak=np.ndarray,
        fft_peak=dict(freq=np.ndarray, amp=np.ndarray),
        fft_bg=dict(freq=np.ndarray, amp=np.ndarray),
        label=str,
      )
    """

    # kept for backward compatibility
    DEFAULT_SAVE_DIR = None

    def __init__(self, style=None, paths=None, default_save_dir=None):
        self.style = style
        self.paths = paths
        self.default_save_dir = None if default_save_dir is None else Path(default_save_dir)

    @classmethod
    def set_default_save_dir(cls, path):
        cls.DEFAULT_SAVE_DIR = None if path is None else Path(path)

    def _resolve_default_save_dir(self) -> Path:
        # 1) instance-level explicit override
        if self.default_save_dir is not None:
            return self.default_save_dir

        # 2) class-level fallback for backward compatibility
        if self.DEFAULT_SAVE_DIR is not None:
            return Path(self.DEFAULT_SAVE_DIR)

        # 3) paths object if provided
        if self.paths is not None:
            return Path(self.paths.analysis_root) / "general_figures"

        raise ValueError(
            "No default save directory configured. "
            "Provide save_dir=..., default_save_dir=..., set "
            "DifferentialFFTMultiPlotter.DEFAULT_SAVE_DIR, or pass paths=..."
        )

    @staticmethod
    def _apply_single_experiment_aesthetics(ax):
        g = globals()
        for fname in (
            "_apply_default_axes_style",
            "_format_axes_default",
            "_format_axes",
            "_apply_axes_style",
            "_apply_grid_style",
        ):
            f = g.get(fname, None)
            if callable(f):
                try:
                    f(ax)
                    return
                except Exception:
                    pass

        ax.grid()

    @staticmethod
    def _get_color_cycle():
        try:
            prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
            if prop_cycle is not None:
                colors = prop_cycle.by_key().get("color", [])
                if colors:
                    return list(colors)
        except Exception:
            pass
        return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    @staticmethod
    def legend_title_default() -> str:
        return globals()["legend_title_default"](scan_type="delay")

    @staticmethod
    def _ensure_legend_clickable(fig, ax, legend, labels, artists_by_label):
        """
        Same strategy as in DifferentialTimeTraceMultiPlotter:
        - try project-level `_make_legend_clickable`
        - fallback to a local pick handler
        """
        used_external = False
        g = globals()
        f = g.get("_make_legend_clickable", None)

        if callable(f):
            try:
                sig = inspect.signature(f)
                params = sig.parameters

                artists_by_label = dict(artists_by_label) if artists_by_label is not None else {}
                legend_handles = list(getattr(legend, "legendHandles", []))
                if not legend_handles:
                    legend_handles = list(legend.get_lines())

                artists_by_legend = {}
                for h, lbl in zip(legend_handles, labels):
                    arts = artists_by_label.get(lbl, [])
                    artists_by_legend[h] = arts

                candidates = {
                    "fig": fig,
                    "ax": ax,
                    "legend": legend,
                    "artists_by_label": artists_by_label,
                    "label_to_artists": artists_by_label,
                    "artists_by_legend": artists_by_legend,
                    "handle_to_artists": artists_by_legend,
                }

                kwargs = {}
                for name, param in params.items():
                    if name in candidates and candidates[name] is not None:
                        kwargs[name] = candidates[name]

                required = [
                    n
                    for n, p in params.items()
                    if p.default is inspect._empty
                    and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                ]

                if all(r in kwargs for r in required):
                    f(**kwargs)
                    used_external = True
            except Exception:
                used_external = False

        if used_external:
            return

        legend_handles = list(getattr(legend, "legendHandles", []))
        if not legend_handles:
            legend_handles = list(legend.get_lines())

        line_to_artists = {}
        for handle, label in zip(legend_handles, labels):
            arts = artists_by_label.get(label, [])
            line_to_artists[handle] = arts
            try:
                handle.set_picker(True)
                handle.set_pickradius(5)
            except Exception:
                try:
                    handle.set_picker(True)
                except Exception:
                    pass

        def on_pick(event):
            legline = event.artist
            if legline not in line_to_artists:
                return
            arts = line_to_artists[legline]
            if not arts:
                return

            try:
                vis = not arts[0].get_visible()
            except Exception:
                vis = True

            for art in arts:
                try:
                    art.set_visible(vis)
                except Exception:
                    pass

            try:
                legline.set_alpha(0.4 if vis else 0.2)
            except Exception:
                pass

            try:
                fig.canvas.draw_idle()
            except Exception:
                fig.canvas.draw()

        try:
            fig.canvas.mpl_connect("pick_event", on_pick)
        except Exception:
            pass

    def define_fig_name_auto(self, series_list):
        name = "FFT_"

        for i in range(len(series_list)):
            exp = series_list[i]["experiment"]

            s_name = exp["sample_name"]
            T_K = exp["temperature_K"]
            ex_wl_nm = exp["excitation_wl_nm"]
            fl_mJ_cm2 = exp["fluence_mJ_cm2"]
            tw_fs = exp["time_window_fs"]
            label = str(exp.get("label", "")).strip()

            if label != "":
                label = f"_label_{label}_"

            dummy = "__" if i < (len(series_list) - 1) else ""

            name += (
                f"Exp{i+1}_{s_name}_{T_K}K_ex_wl_{ex_wl_nm}nm_"
                f"flu_{fl_mJ_cm2}mJcm2_tw_{tw_fs}fs{label}{dummy}"
            )

        return name

    def plot(
        self,
        series_list,
        *,
        time_unit="ps",
        freq_unit="cm^-1",
        xlim_freq=None,
        ylim_freq=None,
        ylim_time=None,
        title=None,
        legend_title=None,
        show=True,
        save=False,
        save_dir=None,
        save_name=None,
        save_format="png",
        save_dpi=300,
        save_overwrite=True,
        legend_outside=True,
    ):
        if legend_title is None:
            legend_title = self.legend_title_default()

        tu = str(time_unit).strip().lower()
        if tu not in ("fs", "ps"):
            raise ValueError("time_unit must be 'fs' or 'ps'.")

        fig, (ax_time, ax_fft) = plt.subplots(2, 1, sharex=False, figsize=(9.0, 7.0))

        self._apply_single_experiment_aesthetics(ax_time)
        self._apply_single_experiment_aesthetics(ax_fft)

        colors = self._get_color_cycle()

        artists_by_label = {}
        legend_handles = []
        legend_labels = []

        fu = str(freq_unit).strip().lower()
        if fu.startswith("cm"):
            xlab_fft = r"Frequency [cm$^{-1}$]"
        elif fu == "thz":
            xlab_fft = "Frequency [THz]"
        else:
            xlab_fft = "Frequency [a.u.]"

        xlabel_time = "Delay [ps]" if tu == "ps" else "Delay [fs]"

        for i, s in enumerate(list(series_list)):
            label = str(s.get("label", f"series_{i+1}")).strip()
            delay_offset_ps = float(s.get("delay_offset_ps", 0.0))

            t_ps = np.asarray(s["time_ps"], float)
            y_peak = np.asarray(s["signal_peak"], float)

            f_peak = np.asarray(s["fft_peak"]["freq"], float)
            a_peak = np.asarray(s["fft_peak"]["amp"], float)
            f_bg = np.asarray(s["fft_bg"]["freq"], float)
            a_bg = np.asarray(s["fft_bg"]["amp"], float)

            if tu == "ps":
                x_time = t_ps + delay_offset_ps
            else:
                x_time = (t_ps + delay_offset_ps) * 1e3

            color = colors[i % len(colors)]

            ln_time = ax_time.plot(
                x_time,
                y_peak,
                "o",
                markersize=PlotStyle.marker_size - 2,
                label=label,
                color=color,
            )[0]

            ln_fft_peak = ax_fft.plot(
                f_peak,
                a_peak,
                color=color,
                linestyle="-",
                label=None,
            )[0]

            ln_fft_bg = ax_fft.plot(
                f_bg,
                a_bg,
                color=color,
                linestyle="--",
                alpha=0.4,
                label=None,
            )[0]

            artists_by_label[label] = [ln_time, ln_fft_peak, ln_fft_bg]
            legend_handles.append(ln_time)
            legend_labels.append(label)

        ax_time.set_ylabel("$\\int$ΔIdq [a.u.]", fontsize=PlotStyle.label_fontsize)
        ax_time.set_xlabel(xlabel_time, fontsize=PlotStyle.label_fontsize)
        ax_fft.set_ylabel("|FFT|", fontsize=PlotStyle.label_fontsize)
        ax_fft.set_xlabel(xlab_fft, fontsize=PlotStyle.label_fontsize)

        if xlim_freq is not None:
            ax_fft.set_xlim(xlim_freq)
        if ylim_freq is not None:
            ax_fft.set_ylim(ylim_freq)
        if ylim_time is not None:
            ax_time.set_ylim(ylim_time)

        if title is not None:
            ax_time.set_title(title, y=1.01)

        legend = None
        if legend_handles:
            if bool(legend_outside):
                legend = ax_time.legend(
                    legend_handles,
                    legend_labels,
                    title=legend_title,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.41),
                    borderaxespad=0.0,
                )
            else:
                legend = ax_time.legend(
                    legend_handles,
                    legend_labels,
                    title=legend_title,
                )

            if legend is not None:
                self._ensure_legend_clickable(
                    fig, ax_time, legend, legend_labels, artists_by_label
                )

        fig.tight_layout()

        saved_path = None
        if bool(save):
            if save_dir is None:
                save_dir = self._resolve_default_save_dir()
            else:
                save_dir = Path(save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)

            if save_name is None or str(save_name).strip() == "":
                save_name = self.define_fig_name_auto(series_list)

            fmt = str(save_format).lstrip(".").lower()
            saved_path = save_dir / f"{save_name}.{fmt}"

            if (not saved_path.exists()) or bool(save_overwrite):
                fig.savefig(str(saved_path), dpi=int(save_dpi), bbox_inches="tight")

        if bool(show):
            plt.show()
        else:
            plt.close(fig)

        return fig, (ax_time, ax_fft), saved_path


class DifferentialFluenceTraceMultiPlotter:
    """
    Multi-experiment fluence-trace plotter for differential integrals.

    Expected per-series dict keys:
      - label (str)
      - fluence_mJ_cm2 (np.ndarray)
      - fluence_offset (float, optional)
      - int_delta (np.ndarray)
      - int_abs_delta (np.ndarray)
      - err_delta (np.ndarray)      # derived from background (signed panel)
      - err_abs_delta (np.ndarray)  # derived from background (abs panel)

    Produces 2 panels:
      top:  ΔI integral
      bot: |ΔI| integral

    Legend is clickable:
      - toggles BOTH panels
      - toggles line + errorbar artists
    """

    # kept for backward compatibility
    DEFAULT_SAVE_DIR = None

    def __init__(self, style=None, paths=None, default_save_dir=None):
        self.style = style
        self.paths = paths
        self.default_save_dir = None if default_save_dir is None else Path(default_save_dir)

    @classmethod
    def set_default_save_dir(cls, path):
        cls.DEFAULT_SAVE_DIR = None if path is None else Path(path)

    def _resolve_default_save_dir(self) -> Path:
        # 1) instance-level explicit override
        if self.default_save_dir is not None:
            return self.default_save_dir

        # 2) class-level fallback for backward compatibility
        if self.DEFAULT_SAVE_DIR is not None:
            return Path(self.DEFAULT_SAVE_DIR)

        # 3) paths object if provided
        if self.paths is not None:
            return Path(self.paths.analysis_root) / "general_figures"

        raise ValueError(
            "No default save directory configured. "
            "Provide save_dir=..., default_save_dir=..., set "
            "DifferentialFluenceTraceMultiPlotter.DEFAULT_SAVE_DIR, or pass paths=..."
        )

    def legend_title_default(self) -> str:
        return legend_title_default(scan_type="fluence")

    # ---------- aesthetics helpers ----------
    def _get_color_cycle(self):
        try:
            cols = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
            if cols:
                return list(cols)
        except Exception:
            pass
        return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    def _apply_single_like_axes_style(self, ax):
        # Try to mimic single-experiment look: grid + minor ticks + ticks-in + full box
        ax.grid()
        try:
            ax.tick_params(which="both", direction="in", top=True, right=True)
        except Exception:
            pass

        try:
            ax.axhline(0.0, lw=1.0, alpha=0.4, zorder=0)
        except Exception:
            pass

    # ---------- legend click handling ----------
    def _call_project_make_legend_clickable(self, legend):
        try:
            _make_legend_clickable(legend)
            return True
        except Exception:
            return False

    def _install_fallback_clickable_legend(self, fig, legend, artists_by_label):
        """
        Always-working fallback: installs a pick_event handler that toggles
        visibility of all artists belonging to a legend entry label.
        """
        if legend is None:
            return

        try:
            handles = list(getattr(legend, "legend_handles", None) or legend.legendHandles)
        except Exception:
            handles = []

        texts = []
        try:
            texts = list(legend.get_texts())
        except Exception:
            texts = []

        n = min(len(handles), len(texts))
        if n == 0:
            return

        handle_to_label = {}
        for i in range(n):
            lbl = texts[i].get_text()
            handle_to_label[handles[i]] = lbl
            try:
                handles[i].set_picker(True)
                handles[i].set_pickradius(6)
            except Exception:
                pass

        for t in texts:
            try:
                t.set_picker(True)
                t.set_pickradius(6)
            except Exception:
                pass

        def _set_entry_alpha(lbl, visible):
            for i in range(n):
                if texts[i].get_text() == lbl:
                    try:
                        texts[i].set_alpha(0.4 if visible else 0.25)
                    except Exception:
                        pass
                    try:
                        handles[i].set_alpha(0.4 if visible else 0.25)
                    except Exception:
                        pass

        def _toggle_label(lbl):
            arts = artists_by_label.get(lbl, [])
            if not arts:
                return

            vis0 = None
            for a in arts:
                try:
                    vis0 = bool(a.get_visible())
                    break
                except Exception:
                    continue
            new_vis = (not vis0) if vis0 is not None else False

            for a in arts:
                try:
                    a.set_visible(new_vis)
                except Exception:
                    pass

            _set_entry_alpha(lbl, new_vis)
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

        def on_pick(event):
            artist = event.artist
            if artist in handle_to_label:
                _toggle_label(handle_to_label[artist])
                return
            if texts and artist in texts:
                try:
                    _toggle_label(artist.get_text())
                except Exception:
                    pass

        try:
            fig.canvas.mpl_connect("pick_event", on_pick)
        except Exception:
            pass

    def define_fig_name_auto(self, series_list):
        name = "Diff_Fluence_Analysis__"

        for i in range(len(series_list)):
            exp = series_list[i]["experiment"]

            s_name = exp["sample_name"]
            T_K = exp["temperature_K"]
            ex_wl_nm = exp["excitation_wl_nm"]
            delay_fs = exp["delay_fs"]
            tw_fs = exp["time_window_fs"]
            label = str(exp.get("label", "")).strip()

            if label != "":
                label = f"_label_{label}_"

            dummy = "__" if i < (len(series_list) - 1) else ""

            name += (
                f"Exp{i+1}_{s_name}_{T_K}K_ex_wl_{ex_wl_nm}nm_"
                f"delay_{delay_fs}fs_tw_{tw_fs}fs{label}{dummy}"
            )

        return name

    # ---------- robust errorbar artist collection ----------
    def _collect_errorbar_artists(self, cont):
        """
        Return a flat list of matplotlib Artist objects created by Axes.errorbar,
        robust across matplotlib versions.
        """
        arts = []

        def _add(obj):
            if obj is None:
                return
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    _add(it)
                return
            if hasattr(obj, "set_visible") and hasattr(obj, "get_visible"):
                arts.append(obj)

        try:
            _add(getattr(cont, "lines", None))
        except Exception:
            pass
        try:
            _add(getattr(cont, "caplines", None))
        except Exception:
            pass
        try:
            _add(getattr(cont, "barlinecols", None))
        except Exception:
            pass

        out = []
        seen = set()
        for a in arts:
            aid = id(a)
            if aid not in seen:
                seen.add(aid)
                out.append(a)
        return out

    # ---------- main plot ----------
    def plot(
        self,
        series_list,
        *,
        fluence_unit="mJ/cm$^2$",
        as_lines=False,
        show_errorbars=True,
        errorbar_scale=1.0,
        title=None,
        legend_title=None,
        show=True,
        save=False,
        save_dir=None,
        save_name=None,
        save_format="png",
        save_dpi=400,
        save_overwrite=True,
        legend_outside=True,
    ):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 7.0))

        if title:
            try:
                ax_top.set_title(title)
            except Exception:
                pass

        self._apply_single_like_axes_style(ax_top)
        self._apply_single_like_axes_style(ax_bot)

        ax_top.set_ylabel(r"$\int$∆Idq [a.u.]", fontsize=PlotStyle.label_fontsize)
        ax_bot.set_ylabel(r"$\int$|∆I|dq [a.u.]", fontsize=PlotStyle.label_fontsize)
        ax_bot.set_xlabel(f"Fluence [{fluence_unit}]", fontsize=PlotStyle.label_fontsize)

        colors = self._get_color_cycle()
        artists_by_label = {}

        for i, s in enumerate(series_list):
            lbl = str(s.get("label", f"series {i+1}"))
            x = np.asarray(s.get("fluence_mJ_cm2", []), float)
            xoff = float(s.get("fluence_offset", 0.0))
            xplot = x + xoff

            y1 = np.asarray(s.get("int_delta", []), float)
            y2 = np.asarray(s.get("int_abs_delta", []), float)

            c = colors[i % len(colors)]
            marker = "o"
            ls = "-" if as_lines else "none"

            e1 = None
            e2 = None
            if show_errorbars:
                e1 = np.asarray(s.get("err_delta", []), float) * float(errorbar_scale)
                e2 = np.asarray(s.get("err_abs_delta", []), float) * float(errorbar_scale)

            these_artists = []

            # --- TOP (ΔI)
            if show_errorbars and e1 is not None and len(e1) == len(y1) and len(y1) == len(xplot):
                cont1 = ax_top.errorbar(
                    xplot, y1, yerr=e1, fmt=marker, linestyle=ls, color=c, label=lbl
                )
                these_artists.extend(self._collect_errorbar_artists(cont1))
            else:
                ln1, = ax_top.plot(xplot, y1, marker=marker, linestyle=ls, color=c, label=lbl)
                these_artists.append(ln1)

            # --- BOTTOM (|ΔI|)
            if show_errorbars and e2 is not None and len(e2) == len(y2) and len(y2) == len(xplot):
                cont2 = ax_bot.errorbar(
                    xplot, y2, yerr=e2, fmt=marker, linestyle=ls, color=c, label=lbl
                )
                these_artists.extend(self._collect_errorbar_artists(cont2))
            else:
                ln2, = ax_bot.plot(xplot, y2, marker=marker, linestyle=ls, color=c, label=lbl)
                these_artists.append(ln2)

            if lbl in artists_by_label:
                artists_by_label[lbl].extend(these_artists)
            else:
                artists_by_label[lbl] = these_artists

        leg_title = legend_title if legend_title is not None else self.legend_title_default()

        if legend_outside:
            leg = ax_top.legend(
                title=leg_title,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
            try:
                fig.subplots_adjust(right=0.48)
            except Exception:
                pass
        else:
            leg = ax_top.legend(title=leg_title)

        self._call_project_make_legend_clickable(leg)
        self._install_fallback_clickable_legend(fig, leg, artists_by_label)

        saved_path = None
        if save:
            if save_dir is None:
                save_dir = self._resolve_default_save_dir()
            else:
                save_dir = Path(save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)

            if save_name is None or str(save_name).strip() == "":
                save_name = self.define_fig_name_auto(series_list)

            fmt = str(save_format).lstrip(".").lower()
            saved_path = save_dir / f"{save_name}.{fmt}"

            if (not bool(save_overwrite)) and saved_path is not None:
                try:
                    if saved_path.exists():
                        base = save_dir / str(save_name)
                        k = 2
                        while (save_dir / f"{save_name}__{k}.{fmt}").exists():
                            k += 1
                        saved_path = save_dir / f"{save_name}__{k}.{fmt}"
                except Exception:
                    pass

            fig.savefig(str(saved_path), dpi=int(save_dpi), bbox_inches="tight")

        if show:
            try:
                plt.show()
            except Exception:
                pass

        return fig, (ax_top, ax_bot), saved_path
    
