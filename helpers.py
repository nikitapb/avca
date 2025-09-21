import pynwb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate

# The following functions can be found in the openscope-databook of The Allen Institute (https://alleninstitute.github.io/openscope_databook, last accessed 20.09.2025)

def get_dff_any(nwb, plane):
    # sometimes varies from plane to plane... 

    if contains_dfOverF(nwb, plane):
        dff_timestamps, dff_trace = get_dfOverF(nwb, plane)
    
    elif contains_dff_timeseries(nwb, plane):
        dff_timestamps, dff_trace = get_dff_timeseries(nwb, plane)
    else:
        dff_timestamps, dff_trace = get_dff(nwb, plane)

    return dff_timestamps, dff_trace


def contains_dfOverF(nwb: pynwb.file.NWBFile, recording_plane: str):
    return "DfOverF" in nwb.processing[recording_plane].data_interfaces.keys()


def contains_dff_timeseries(nwb: pynwb.file.NWBFile, recording_plane: str):
    return "dff_timeseries" in nwb.processing[recording_plane].data_interfaces.keys()

def get_dfOverF(nwb: pynwb.file.NWBFile, recording_plane: str):
    dff = nwb.processing[recording_plane].data_interfaces["DfOverF"].roi_response_series["deltaFoverF"]
    return dff.timestamps, dff.data

def get_dff(nwb: pynwb.file.NWBFile, recording_plane: str):
    dff = nwb.processing[recording_plane].data_interfaces["dff"].roi_response_series["dff_timeseries"]
    return dff.timestamps, dff.data

def get_dff_timeseries(nwb: pynwb.file.NWBFile, recording_plane: str):
    dff = nwb.processing[recording_plane].data_interfaces["dff_timeseries"].roi_response_series["dff_timeseries"] #.roi_response_series["dff_timeseries"]
    return dff.timestamps, dff.data

def interpolate_dff(dff_trace, timestamps, interp_hz):

    # generate regularly-space x values and interpolate along it
    time_axis = np.arange(timestamps[0], timestamps[-1], step=(1/interp_hz))
    interp_dff = []
    
    # interpolate channel by channel to save RAM
    for channel in range(dff_trace.shape[1]):
        f = interpolate.interp1d(timestamps, dff_trace[:,channel], axis=0, kind="nearest", fill_value="extrapolate")
        interp_dff.append(f(time_axis))
    
    interp_dff = np.array(interp_dff)

    return interp_dff.T #to keep same shape as dff_trace


def get_zscore(window, stim_idx, i_ax=1):
    baseline_means = np.nanmean(window[:,:stim_idx], axis=i_ax)
    stdevs = np.std(window, axis=i_ax)
    
    baseline_means = np.expand_dims(baseline_means, 1)
    stdevs = np.expand_dims(stdevs, 1)

    zscore_window = (window - baseline_means) / stdevs
    return zscore_window

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#The following functions were created for the thesis

def getStimDurations(nwb):
    stim_keys = list(nwb.intervals.keys())
    durations_each = []

    for stim_key in stim_keys:
        durations = []
        stim_table = nwb.intervals[stim_key][:]
        for i in range(len(stim_table)):
            durations.append(stim_table.at[i, "stop_time"] - stim_table.at[i, "start_time"])
        durations_each.append({'stimulus': stim_key, 'mean_presentation_duration': np.mean(durations), 'Unit': 's'})
    return pd.DataFrame(durations_each)
    
def getRFcoords(xpos, ypos):
    xpos = float(xpos)
    xpos /= 10
    xpos += 4

    ypos = float(ypos)
    ypos /= 10
    ypos -= 4
    ypos *= (-1)
    return int(xpos), int(ypos)

def getDisplayMask(stim_table):
    display_mask = np.zeros((9,9), dtype=float)
    for j in range(stim_table.shape[0]):

        x, y = getRFcoords(stim_table.iloc[j]["x_position"], stim_table.iloc[j]["y_position"])
        display_mask[y, x] += 1

    return display_mask

def getRegionByPlane(plane):
    return {
    'VISam_6': 'AM #1',
    'VISam_7': 'AM #2',
    'VISp_0': 'V1 Anterior #1',
    'VISp_1': 'V1 Anterior #2',
    'VISp_2': 'V1 Posterior #1',
    'VISp_3': 'V1 Posterior #2',
    'VISp_4': 'V1 Center #1',
    'VISp_5': 'V1 Center #2',
    }[plane]

def getStimName(stimKey):
    return {
    "drifting_gratings_field_block_presentations_vsync": "Drifting Gratings",
    "homogeneous_background_presentations_vsync": "Homogeneous Background",
    "rdkCircle_presentations_vsync": "Random Dot Kinematogram", #(Center)
    "rdkSqr_presentations_vsync": "random dot kinematogram outside",
    "receptive_field_block_presentations_vsync": "Gabor Stimuli",
    "sparse_noise_8x14_presentations_vsync": "Sparse Noise"
    }[stimKey]
