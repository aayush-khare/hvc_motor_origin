import streamlit as st
import scipy.io
from scipy import signal
import numpy as np
from collections import defaultdict
import glob
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TraceViewer:
    
    def __init__(self, folder_path='../RAtraces', file_pattern='*.mat'):
        """
        Initialize the trace viewer
        
        Args:
            folder_path: Path to folder containing .mat files
            file_pattern: Pattern to match .mat files (e.g., 'traces_*.mat')
        """
        self.folder_path = folder_path
        self.file_pattern = file_pattern
        self.files = self.get_mat_files()
        self.current_file_index = 0
        self.current_data = None

        if not self.files:
            st.warning(f"No .mat files found matching pattern '{file_pattern}' in '{folder_path}'")
            return
        
        st.success(f"Found {len(self.files)} .mat files")
        self.load_current_file()
        
    def get_mat_files(self):
        """Get all .mat files matching the pattern"""
        pattern = os.path.join(self.folder_path, self.file_pattern)
        files = glob.glob(pattern)
        files.sort()
        return files
      
    def load_current_file(self):
        """Load the current .mat file and extract traces"""
        if not self.files:
            return
        
        current_file = self.files[self.current_file_index]
        try:
            mat_data = scipy.io.loadmat(current_file)
            
            if 'traces' in mat_data:
                self.current_data = mat_data['traces']
            else:
                available_vars = [key for key in mat_data.keys() if not key.startswith('__')]
                st.error(f"No 'traces' variable found. Available: {available_vars}")
                self.current_data = None
                
        except (FileNotFoundError, OSError, ValueError) as e:
            st.error(f"Error loading file: {e}")
            self.current_data = None
    
    def load_file_by_index(self, index):
        """Load a specific file by index"""
        if 0 <= index < len(self.files):
            self.current_file_index = index
            self.load_current_file()
    
    def get_current_filename(self):
        """Get the current filename"""
        if self.files:
            return os.path.basename(self.files[self.current_file_index])
        return None
    
    def get_trace_data(self, trace_index):
        """Get data for a specific trace"""
        if self.current_data is not None and trace_index < self.current_data.shape[1]:
            return self.current_data[0, trace_index].flatten()
        return None
    
    def get_num_traces(self):
        """Get number of traces in current file"""
        if self.current_data is not None:
            return self.current_data.shape[1]
        return 0
    
def smoothen_trace(trace, smooth_window_ms, polyorder, sampling_rate=40000):
    """
    Smoothen a trace by using the savgol filter and a selected smoothing window and polynomial order.
    Depending on the smoothing window, high frequency noise can be reduced and make 
    subthreshold events more visible. By default, the savgol filter is used with a window of 1 ms 
    and polynomial order of 0, corresponding to a moving average filter.

    Args:
        trace: 1D numpy array of trace data
        smooth_window_ms: Smoothing window in milliseconds
        polyorder: Polynomial order for savgol filter
        sampling_rate: Sampling rate in Hz (default: 40000)
        spike_threshold: Threshold for spike detection (default: 20.0)
    Returns:
        smoothened trace as 1D numpy array
    """
    window_samples = int(sampling_rate * smooth_window_ms / 1000)
    
    smoothed_trace = signal.savgol_filter(trace, window_length=window_samples | 1, polyorder=polyorder)    
    trace_clean = np.copy(smoothed_trace)
    
    return trace_clean

def plot_traces(viewer, selected_traces, smoothen, smooth_window_ms, polyorder, plot_type='overlay', sampling_rate_analysis=10000):
    """
    Plot selected traces using Plotly
    
    Args:
        viewer: TraceViewer instance
        selected_traces: List of trace indices to plot
        smoothen: condition to smoothen traces for visualization
        smooth_window_ms: Smoothing window in milliseconds
        polyorder: Polynomial order for savgol filter
        plot_type: 'overlay' or 'subplots'
        sampling_rate_analysis: Sampling rate in Hz (default: 10000)
    Returns:
        None
    """

    x_label = None
    y_label = None

    sampling_rate_recording = 40000
    downsample_factor = sampling_rate_recording // sampling_rate_analysis

    if not selected_traces:
        st.warning("No traces selected for visualization")
        return
    
    # Either overlay all traces or create subplots for each trace
    if plot_type == 'overlay':
        fig = go.Figure()
        
        for idx in selected_traces:
            trace_data = viewer.get_trace_data(idx)
            if trace_data is not None:
                x = np.arange(len(trace_data)) * 1000 / (sampling_rate_recording)

                x_label = 'Time (ms)'
                y_label = 'Membrane Potential (mV)'
                
                if smoothen:
                    trace_data = smoothen_trace(trace_data, smooth_window_ms, polyorder, sampling_rate=sampling_rate_recording)
                    #baseline = np.median(trace_data[trace_data <= np.percentile(trace_data, 10)])
                    #spike_threshold = 20.0
                    #spike_indices = np.where(trace_data > baseline + spike_threshold)[0]
                    #if spike_indices.size != 0:
                        #start = spike_indices[0]
                    
                        # take the trace - 15 ms before the spike start and + 30 ms after spike start
                        #trace_mask = (np.arange(len(trace_data)) <= start - int(0.015 * sampling_rate_recording)) | (np.arange(len(trace_data)) >= start + int(0.030 * sampling_rate_recording))
                        # get the mean of the trace excluding the spike region
                        #start_time = (start - int(0.015 * sampling_rate_recording)) * 1000 / (sampling_rate_recording)
                        #end_time = (start + int(0.030 * sampling_rate_recording)) * 1000 / (sampling_rate_recording)
                        #st.text(f'{start_time}, {end_time}')
                        #st.text(f'Trace statistics after removing spike region for trace {idx + 1}:')
                        #st.text(f'{trace_data[trace_mask].mean(), trace_data[trace_mask].std()}')
                    #else:

                        #st.text(f'Trace statistics (no spikes detected) for trace {idx + 1}:')
                        #st.text(f'{trace_data.mean(), trace_data.std()}')
                fig.add_trace(go.Scatter(
                    x=x[::downsample_factor],
                    y=trace_data[::downsample_factor],
                    mode='lines',
                    name=f'Trace {idx + 1}',
                    line=dict(width=1.5)
                ))
        
        fig.update_layout(
            title=f"Neural Traces - {viewer.get_current_filename()}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode='x unified',
            height=600,
            yaxis_range=[-90.0, 20.0],
        )
        
    else:  # subplots
        n_traces = len(selected_traces)
        fig = make_subplots(
            rows=n_traces, 
            cols=1,
            subplot_titles=[f'Trace {idx + 1}' for idx in selected_traces],
            vertical_spacing=0.05
        )
        
        for i, idx in enumerate(selected_traces, 1):
            trace_data = viewer.get_trace_data(idx)
            if trace_data is not None:
                x = np.arange(len(trace_data)) * 1000 / (sampling_rate_recording)
                x_label = 'Time (ms)'
                y_label = 'Membrane Potential (mV)'
                
                if smoothen:
                    trace_data = smoothen_trace(trace_data, smooth_window_ms, polyorder, sampling_rate=sampling_rate_recording)

                fig.add_trace(
                    go.Scatter(x=x, y=trace_data, mode='lines', 
                              name=f'Trace {idx + 1}', line=dict(width=1)),
                    row=i, col=1
                )
        
        fig.update_xaxes(title_text=x_label, row=n_traces, col=1)
        fig.update_yaxes(title_text=y_label)
        fig.update_layout(
            title=f"Neural Traces - {viewer.get_current_filename()}",
            height=300 * n_traces,
            showlegend=False
        )
    
    st.plotly_chart(fig)

def analyze_traces(viewer, file_index, selected_traces, smoothen, smooth_window_ms, polyorder, sampling_rate_analysis=10000):
    """
    Analyze selected traces for subthreshold events
    
    Args:
        viewer: TraceViewer instance
        file_index: Index of the file (neuron) being analyzed
        selected_traces: List of trace indices to plot
        smoothen: Condition to smoothen traces for analysis
        smooth_window_ms: Smoothing window in milliseconds
        polyorder: Polynomial order for savgol filter
        sampling_rate_analysis: Sampling rate in Hz (default: 10000)
    Returns:
        None
    """

    min_event_height = 2.0 # mV
    max_event_height = 25.0 # mV
    window_size_ms = 50.0
    step_size_ms = 10.0
    #time_window_ms = 5.0
    time_window_grouping_ms = 5.0
    shift_from_peak_for_recalculating_baseline_ms = 0.0
    width_window_for_recalculating_baseline_ms = 10.0
    
    sampling_rate_recording = 40000
    downsample_factor = sampling_rate_recording // sampling_rate_analysis

    window_samples = int(window_size_ms * sampling_rate_analysis / 1000)
    step_samples = int(step_size_ms * sampling_rate_analysis / 1000)

    if not selected_traces:
        st.warning("No traces selected")
        return
    
    results = {}

    for idx in selected_traces:

        trace_index = idx + 1
        trace_data = viewer.get_trace_data(idx)
        if trace_data is not None:
                       
            if smoothen:
                trace_data = smoothen_trace(trace_data, smooth_window_ms, polyorder, sampling_rate=sampling_rate_recording)

            all_events = []
            event_local_baselines = []
            trace_data = trace_data[::downsample_factor]
            baseline = np.median(trace_data[trace_data <= np.percentile(trace_data, 5)]) # global baseline

            # sliding window peak detection, this captures any local variations in baseline, followed by recalculation of baseline before each detected peak
            # will detect a peak multiple times if it appears in multiple windows, will be filtered later
            for start_idx in range(0, len(trace_data), step_samples):
                end_idx = min(start_idx + window_samples, len(trace_data))

                window_data = trace_data[start_idx:end_idx]

                window_baseline = np.median(window_data[window_data <= np.percentile(window_data, 5)])
                # local baseline as weighted average of window baseline and global baseline
                local_baseline = window_baseline * 0.7 + baseline * 0.3

                local_threshold = local_baseline + min_event_height

                peak_indices, _ = signal.find_peaks(
                    window_data,
                    height=local_threshold,
                    prominence=1.0
                )

                for peak_idx in peak_indices:
                    global_peak_idx = peak_idx + start_idx
                    if global_peak_idx < len(trace_data):
                        event_amplitude = trace_data[global_peak_idx] - local_baseline
                        event_peak_value = trace_data[global_peak_idx]
                        
                        if event_amplitude < max_event_height:
                            
                            local_window_start_idx = global_peak_idx - int(sampling_rate_analysis * (shift_from_peak_for_recalculating_baseline_ms + width_window_for_recalculating_baseline_ms) * 0.001)  # 20ms before peak
                            local_window_end_idx = global_peak_idx - int(sampling_rate_analysis * (shift_from_peak_for_recalculating_baseline_ms) * 0.001)
                            if local_window_start_idx < 0:
                                local_window_start_idx = 0
                            if local_window_end_idx < 0:
                                pre_event_baseline = local_baseline
                                recalculated_baseline_time = 0.0
                            else:
                                local_window_data = trace_data[local_window_start_idx:local_window_end_idx]
                                pre_event_baseline = np.median(local_window_data[local_window_data <= np.percentile(local_window_data, 5)])
                                
                                baseline_mask = local_window_data <= np.percentile(local_window_data, 5)
                                baseline_time_points = np.where(baseline_mask)[0] + local_window_start_idx
                                if len(baseline_time_points) > 0:
                                    baseline_times_ms = baseline_time_points / sampling_rate_analysis * 1000
                                    recalculated_baseline_time = np.median(baseline_times_ms)
                                else:
                                    recalculated_baseline_time = (local_window_start_idx / sampling_rate_analysis) * 1000

                            event_amplitude = trace_data[global_peak_idx] - pre_event_baseline
                            rise_time = (global_peak_idx / sampling_rate_analysis) * 1000 - recalculated_baseline_time
                         
                            if event_amplitude > min_event_height and event_amplitude < max_event_height:
                                all_events.append({
                                    'index': global_peak_idx,
                                    'time_ms': (global_peak_idx / sampling_rate_analysis) * 1000,
                                    'rise_time_ms': rise_time,
                                    'amplitude': event_amplitude,
                                    'local_baseline': pre_event_baseline,
                                    'peak_value': event_peak_value
                                })
                                event_local_baselines.append(pre_event_baseline)

            all_events.sort(key=lambda x: x['index'])
            unique_events = []
            
            # a peak will be detected multiple times in overlapping windows, and with same amplitudes
            # to get unique events, we check all events that have the same 'time_ms' and keep only one of them
            for event in all_events:
                if not unique_events or event['time_ms'] != unique_events[-1]['time_ms']:
                    unique_events.append(event)

            num_events = len(unique_events)
            event_times = [event['time_ms'] for event in unique_events]
            event_rise_times = [event['rise_time_ms'] for event in unique_events]
            event_indices = [event['index'] for event in unique_events]
            event_amplitudes = [event['amplitude'] for event in unique_events]
            event_peak_values = [event['peak_value'] for event in unique_events]
            event_local_baselines = [event['local_baseline'] for event in unique_events]

            results[trace_index] = {
                'event_stats': {
                    'num_events': num_events,
                    'event_times': event_times,
                    'event_rise_times': event_rise_times,
                    'event_indices': event_indices,
                    'event_amplitudes': event_amplitudes.tolist() if hasattr(event_amplitudes, 'tolist') else list(event_amplitudes),
                    'event_local_baselines': event_local_baselines,
                    'event_peak_values': event_peak_values,
                    'median_amplitude': np.median(event_amplitudes) if len(event_amplitudes) > 0 else np.nan,
                }
            }

    results_to_group = prepare_results_for_grouping(results)
    result_dict = {file_index + 1: results_to_group}

    for neuron in result_dict:
        grouped_results = group_events_by_time(result_dict[neuron], time_threshold=time_window_grouping_ms)
    
    st.header('Subthreshold Event Analysis Results')
    st.text('Below are the grouped subthreshold events, returned as lists of tuples where each tuple is (trace index, ' \
            'time of event (ms), local baseline (mV), peak value (mV), event amplitude from local baseline (mV)). ' \
            'Only events with amplitude >= 2.0 mV are considered. ' \
            'If no subthreshold events are detected in a trace, it is represented as (-1, -1, -1, -1, -1). ' \
            'Each group of events is displayed on a new line. ' \
            'Trace indices are 0-based.')
    st.text('This is only the first step of the analysis, further steps involve manually verifying these events, as for ' \
            'few traces, multiple events in close proximity may be detected, and by manual verification with the traces '
            'a single event can be chosen based on either which event occured first or which event resulted in the '
            'largest peak value of the membrane potential. Once this decision is made, the final list of events for ' \
            'each neuron can be passed on to a dictionary for further analysis.')
    
    st.subheader(f'Subthreshold events detected for Neuron {file_index + 1}: ')

    # the grouped results contain events from all traces that are close in time
    # sometimes for a given time point, there may not be events close to it on either side of the time point
    # so these events need to be exluded from the final display
    
    filtered_grouped_results = []
    for group in grouped_results:
        a = sorted(group, key=lambda x: x[0])
        if len(a) > len(selected_traces) // 2:
            filtered_grouped_results.append(group)
        else:
            if filtered_grouped_results:
                prev_group = filtered_grouped_results[-1]
                prev_time_median = np.median([e[1] for e in prev_group])
                curr_time = a[0][1]
                if (abs(curr_time - prev_time_median) <= time_window_grouping_ms):
                    filtered_grouped_results.append(group)
    for group in filtered_grouped_results:
        
        filtered_group = filter_group(group)

        a = sorted(filtered_group, key=lambda x: x[0])
        
        if filtered_grouped_results.index(group) == len(filtered_grouped_results) - 1:
            st.text(f'{a}')
        else:
            st.text(f'{a},')

def filter_group(group):
    """
    If multiple events from the same trace are present in a group, keep only the event with the highest peak value.
    """

    best_by_index = defaultdict()
    for event in group:
        trace_id = event[0]
        peak_value = event[4]

        if trace_id not in best_by_index or peak_value > best_by_index[trace_id][4]:
            best_by_index[trace_id] = event
    
    filtered_group = list(best_by_index.values())
    return filtered_group

def prepare_results_for_grouping(results):
    """
    Prepare results for grouping by extracting relevant event information.
    Args:
        results: Dictionary of results from analyze_traces function
    Returns:
        List of lists, where each inner list contains tuples of
        (time_of_event, rise_time, baseline, peak_value, event_amplitude) for each trace
    """


    neuron_results = []
    for trace_num in sorted(results.keys()):
        data = results[trace_num]
        events = data['event_stats']
        event_amps = events['event_amplitudes']
        event_peak_vals = events['event_peak_values']
        event_local_base = events['event_local_baselines']
        trace_event_list = []
        if events['num_events'] > 5:
            event_times = events['event_times']
            event_rise_times = events['event_rise_times']
            event_times = [round(t, 3) for t in event_times]
            event_rise_times = [round(t, 3) for t in event_rise_times]
            event_amps = [round(t, 3) for t in event_amps]
            event_peak_vals = [round(t, 3) for t in event_peak_vals]
            event_local_base = [round(t, 3) for t in event_local_base]


            for i in range(len(event_times)):
                trace_event_list.append((event_times[i], event_rise_times[i], event_local_base[i], event_peak_vals[i], event_amps[i]))
                    
        else:
            trace_event_list.append((-1, -1, -1, -1, -1))

            
        neuron_results.append(trace_event_list)
    return neuron_results

def group_events_by_time(traces, time_threshold=10.0):
    """
    Groups events from multiple traces based on time proximity.
    
    Args:
        traces: List of lists, where each inner list contains tuples of 
            (time_of_event, baseline, peak_value, event_amplitude)
        time_threshold: Maximum time difference to consider events as grouped together
    
    Returns:
        List of lists, where each inner list contains tuples of 
        (trace_index, time_of_event, baseline, peak_value, event_amplitude) for events 
        that occur within the time threshold
    """
    
    all_events = []
    for trace_id, trace in enumerate(traces):
        for sub_event in trace:
            event_time, event_rise_time, baseline_value, peak_value, event_amp = sub_event
            all_events.append((trace_id, event_time, event_rise_time, baseline_value, peak_value, event_amp))

    all_events.sort(key=lambda x: x[1]) # sorting by time of event
    
    grouped_events = []
    current_group = []
    
    for sub_event in all_events:
        if not current_group:

            current_group.append(sub_event)
        else:
            
            dyn_median = np.median([e[1] for e in current_group])

            time_diff = abs(sub_event[1] - dyn_median)
            if time_diff <= time_threshold:
                current_group.append(sub_event)
            else:

                if current_group:
                    grouped_events.append(current_group)
                current_group = [sub_event]
    

    if current_group:
        grouped_events.append(current_group)
    
    return grouped_events

def main():

    st.set_page_config(page_title="HVC Neuron Trace Viewer", layout="wide")
    st.title("HVC Neuron Trace Viewer")
    
    neuron_class = st.sidebar.selectbox('Neuron Class', ['HVC(RA)', 'HVC(X)'])

    if neuron_class == 'HVC(RA)':
        folder = '../RAtraces'
    else:
        folder = '../Xtraces'

    with st.sidebar:
        st.header("Configuration")
        folder_path = folder
        file_pattern = '*.mat'
        
        if st.button("Load Files"):
            st.session_state.viewer = TraceViewer(folder_path, file_pattern)
    
    if 'viewer' not in st.session_state:
        st.session_state.viewer = TraceViewer(folder_path='../RAtraces', file_pattern='*.mat')
    
    viewer = st.session_state.viewer
    
    if not viewer.files:
        st.info("Configure the folder path and pattern in the sidebar, then click 'Load Files'")
        return
    
    st.subheader("File Selection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        file_options = [os.path.basename(f) for f in viewer.files]
        selected_file = st.selectbox(
            "Select File",
            options=range(len(file_options)),
            format_func=lambda x: f"{x + 1}. {file_options[x]}"
        )
        
        if selected_file != viewer.current_file_index:
            viewer.load_file_by_index(selected_file)
    
    with col2:
        st.metric("Total Files", len(viewer.files))
        st.metric("Current File", selected_file + 1)
    
    if viewer.current_data is not None:
        num_traces = viewer.get_num_traces()
        st.info(f"File contains {num_traces} traces")

        if num_traces > 10:
            num_traces_to_analyze = st.number_input("File contains more than 10 traces, choose the number of traces " \
            "you wish to visualize and analyze", min_value=10, max_value=num_traces, value=min(num_traces, 10))
        else:
            num_traces_to_analyze = num_traces

        st.subheader("Trace Selection & Plotting")
        
        col1, col2 = st.columns(2)
        
        with col1:

            select_all = st.checkbox("Select All Traces (Would be ideal if there are less than 10 traces)", value=False)
            if select_all:
                selected_traces = list(range(num_traces_to_analyze))
            else:
                selected_traces = st.multiselect(
                    "Select Traces to Plot",
                    options=range(num_traces_to_analyze),
                    format_func=lambda x: f"Trace {x + 1}",
                    default=[0] if num_traces_to_analyze > 0 else []
                )
        
        with col2:
            plot_type = st.radio("Plot Type", options=['overlay', 'subplots'])
        
        col3, col4 = st.columns(2)
        with col3:
            sampling_rate_analysis = st.selectbox("Choose a sampling rate for analysis (Hz), " \
            "the recording sampling rate is 40000 Hz", [10000, 20000, 40000])
        with col4:
            select_plot_smoothening = st.selectbox('Smoothen trace? (uses the Savgol filter from scipy.signal)', ['No', 'Yes'])
        col5, col6 = st.columns(2)
        if select_plot_smoothening == 'Yes':
            with col5:
                smooth_window_ms = st.number_input('Smoothing Window (ms)', min_value=0.25, max_value=5.0, value=1.0, step=0.25, format="%.3f")
            with col6:
                polyorder = st.number_input('Polynomial Order for Smoothing', min_value=0, max_value=10, value=0, step=1)  
        else:
            smooth_window_ms = None
            polyorder = None

        if select_plot_smoothening == 'No':
            smoothen = False
            plot_traces(viewer, selected_traces, smoothen, smooth_window_ms, polyorder, plot_type, sampling_rate_analysis)
        elif select_plot_smoothening == 'Yes':
            smoothen = True
            plot_traces(viewer, selected_traces, smoothen, smooth_window_ms, polyorder, plot_type, sampling_rate_analysis)
        
        selected_traces_for_analysis = list(range(num_traces_to_analyze)) 
        if select_plot_smoothening == 'No':
            smoothen = False
            analyze_traces(viewer, viewer.current_file_index, selected_traces_for_analysis, smoothen, smooth_window_ms, polyorder, sampling_rate_analysis)
        elif select_plot_smoothening == 'Yes':
            smoothen = True
            analyze_traces(viewer, viewer.current_file_index, selected_traces_for_analysis, smoothen, smooth_window_ms, polyorder, sampling_rate_analysis)
        
if __name__ == "__main__":
    main()