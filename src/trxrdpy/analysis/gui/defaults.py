"""
Default GUI values copied from the legacy analysis GUI.
"""

DEFAULT_FIT_PEAK_SPECS = {
    "104": {"q_fit_range": (2.20, 2.40), "eta": 0.3},
    "110": {"q_fit_range": (2.40, 2.65), "eta": 0.3},
    "116": {"q_fit_range": (3.577, 3.823), "eta": 0.3},
    "300": {"q_fit_range": (4.3, 4.46), "eta": 0.3},
}


DEFAULT_AZIM_WINDOWS = [
    (-90, 90),
    (-75, -45),
    (-45, -15),
    (-15, 15),
    (15, 45),
    (45, 75),
]


DEFAULT_MULTI_EXPERIMENTS_FIT = [
    dict(
        sample_name="DET70",
        temperature_K=110,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=5,
        time_window_fs=250,
        phi_mode="phi_avg",
        delay_offset_ps=9.6,
        ref_type="dark",
        ref_value=[1466558],
        label=r"V$_2$O$_3$, 110, 1500, 5",
        delay_for_norm_max=40,
    ),
    dict(
        sample_name="DET70",
        temperature_K=110,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=9,
        time_window_fs=250,
        phi_mode="phi_avg",
        delay_offset_ps=9.6,
        ref_type="dark",
        ref_value=[1466588],
        label=r"V$_2$O$_3$, 110, 1500, 9",
        delay_for_norm_max=40,
    ),
    dict(
        label=r"V$_2$O$_3$, 110, 1500, 12",
        merge=[
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=12,
                time_window_fs=250,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466557],
                delay_for_norm_max=40,
            ),
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=12,
                time_window_fs=40,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466584],
                delay_for_norm_max=40,
            ),
        ],
        delay_for_norm_max=40,
    ),
    dict(
        label=r"V$_2$O$_3$, 110, 1500, 25",
        merge=[
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=25,
                time_window_fs=250,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466556],
                delay_for_norm_max=40,
            ),
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=25,
                time_window_fs=40,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466583],
                delay_for_norm_max=40,
            ),
        ],
        delay_for_norm_max=40,
    ),
]


DEFAULT_DIFF_PEAK_SPECS = {
    "012": {"q_range": (1.6438, 1.8), "bg_side": "right"},
    "104": {"q_range": (2.21, 2.40), "bg_side": "left"},
    "110": {"q_range": (2.45, 2.6), "bg_side": "right"},
    "116": {"q_range": (3.58, 3.82), "bg_side": "right"},
    "300": {"q_range": (4.30, 4.46), "bg_side": "left"},
}


DEFAULT_MULTI_EXPERIMENTS_DIFF = [
    dict(
        sample_name="DET70",
        temperature_K=110,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=1.5,
        time_window_fs=250,
        phi_mode="phi_avg",
        delay_offset_ps=9.6,
        ref_type="dark",
        ref_value=[1466560],
        label=r"V$_2$O$_3$, 110, 1500, 1.5",
        delay_for_norm_max=40,
    ),
    dict(
        label=r"V$_2$O$_3$, 110, 1500, 2.4",
        merge=[
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=2.4,
                time_window_fs=250,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466559],
                delay_for_norm_max=40,
            ),
            dict(
                sample_name="DET70",
                temperature_K=110,
                excitation_wl_nm=1500,
                fluence_mJ_cm2=2.4,
                time_window_fs=40,
                phi_mode="phi_avg",
                delay_offset_ps=9.6,
                ref_type="dark",
                ref_value=[1466586],
                delay_for_norm_max=40,
            ),
        ],
        delay_for_norm_max=40,
    ),
]