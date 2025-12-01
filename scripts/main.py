import streamlit as st
import scipy.io
from scipy import signal
import numpy as np
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
                
        except Exception as e:
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

def preprocess_trace(trace, smooth_window_ms, polyorder, sampling_rate=40000, spike_threshold=20.0):
    """
    Preprocess a trace by smoothing using the savgol filter and a selected smoothing window and polynomial order.
    Depending on the smoothing window, high frequency noise can be reduced, spikes may be flattened, and make 
    subthreshold events more visible.

    Args:
        trace: 1D numpy array of trace data
        smooth_window_ms: Smoothing window in milliseconds
        polyorder: Polynomial order for savgol filter
        sampling_rate: Sampling rate in Hz (default: 40000)
        spike_threshold: Threshold for spike detection (default: 20.0)
    Returns:
        Preprocessed trace as 1D numpy array
    """
    window_samples = int(sampling_rate * smooth_window_ms / 1000)
    #if window_samples < 3:
    #    window_samples = 3  # minimum
    
    smoothed_trace = signal.savgol_filter(trace, window_length=window_samples | 1, polyorder=polyorder)
    baseline = np.median(smoothed_trace[smoothed_trace <= np.percentile(smoothed_trace, 10)])
    
    trace_clean = np.copy(smoothed_trace)
    #spike_indices = np.where(trace_clean > baseline + spike_threshold)[0]
    #trace_clean[spike_indices] = baseline + spike_threshold   
    
    return trace_clean

def plot_traces(viewer, selected_traces, preprocess, smooth_window_ms, polyorder, plot_type='overlay', sampling_rate=40000):
    """
    Plot selected traces using Plotly
    
    Args:
        viewer: TraceViewer instance
        selected_traces: List of trace indices to plot
        preprocess: condition to smoothen traces for visualization
        smooth_window_ms: Smoothing window in milliseconds
        polyorder: Polynomial order for savgol filter
        plot_type: 'overlay' or 'subplots'
        sampling_rate: Sampling rate in Hz (default: 40000)
    Returns:
        None
    """

    x_label = None
    y_label = None
    if not selected_traces:
        st.warning("No traces selected")
        return
    
    if plot_type == 'overlay':
        fig = go.Figure()
        
        for idx in selected_traces:
            trace_data = viewer.get_trace_data(idx)
            if trace_data is not None:
                x = np.arange(len(trace_data)) * 1000 / sampling_rate
                x_label = 'Time (ms)'
                y_label = 'Membrane Potential (mV)'
                
                if preprocess:
                    trace_data = preprocess_trace(trace_data, smooth_window_ms, polyorder, sampling_rate=sampling_rate)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=trace_data,
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
                x = np.arange(len(trace_data)) * 1000 / sampling_rate
                x_label = 'Time (ms)'
                y_label = 'Membrane Potential (mV)'
                
                if preprocess:
                    trace_data = preprocess_trace(trace_data, smooth_window_ms, polyorder, sampling_rate=sampling_rate)

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

def analyze_traces(viewer, file_index, selected_traces, preprocess, smooth_window_ms, polyorder, sampling_rate=40000):
    """
    Analyze selected traces for subthreshold events
    
    Args:
        viewer: TraceViewer instance
        file_index: Index of the file (neuron) being analyzed
        selected_traces: List of trace indices to plot
        preprocess: Condition to smoothen traces for analysis
        smooth_window_ms: Smoothing window in milliseconds
        polyorder: Polynomial order for savgol filter
        sampling_rate: Sampling rate in Hz (default: 40000)
    Returns:
        None
    """

    min_event_height = 1.0
    window_size_ms = 50.0
    step_size_ms = 10.0

    window_samples = int(window_size_ms * sampling_rate / 1000)
    step_samples = int(step_size_ms * sampling_rate / 1000)

    if not selected_traces:
        st.warning("No traces selected")
        return
    
    results = {}

    for idx in selected_traces:

        trace_index = idx + 1
        trace_data = viewer.get_trace_data(idx)
        if trace_data is not None:            
            if preprocess:
                trace_data = preprocess_trace(trace_data, smooth_window_ms, polyorder, sampling_rate=sampling_rate)

            all_events = []
            event_local_baselines = []

            baseline = np.median(trace_data[trace_data <= np.percentile(trace_data, 10)]) # global baseline

            for start_idx in range(0, len(trace_data) - window_samples, step_samples):
                end_idx = start_idx + window_samples
                window_data = trace_data[start_idx:end_idx]

                #local_baseline = np.median(window_data[window_data <= np.percentile(window_data, 10)])
                local_baseline = baseline

                local_threshold = local_baseline + min_event_height
                
                peak_indices, _ = signal.find_peaks(
                    window_data,
                    height=local_threshold,
                    prominence=0.5,
                    distance=int(sampling_rate * 5 * 0.001)  # 5ms minimum distance between peaks
                )
                 
                for peak_idx in peak_indices:
                    global_peak_idx = peak_idx + start_idx
                    if global_peak_idx < len(trace_data):
                        event_amplitude = trace_data[global_peak_idx] - local_baseline
                        event_peak_value = trace_data[global_peak_idx]
                        
                        if trace_data[global_peak_idx] > baseline and event_amplitude >= 1.0 and local_baseline + event_amplitude > trace_data.mean() + 0.5 and event_amplitude < 30.0:
                            
                            local_window_start_idx = global_peak_idx - 600
                            if local_window_start_idx < 0:
                                local_window_start_idx = 0
                            local_window_data = trace_data[local_window_start_idx:global_peak_idx]

                            recalculated_baseline = np.median(local_window_data[local_window_data <= np.percentile(local_window_data, 10)])
                            local_baseline = recalculated_baseline
                            event_amplitude = trace_data[global_peak_idx] - local_baseline
                         
                            if event_amplitude > 0.5 * min_event_height:
                                all_events.append({
                                    'index': global_peak_idx,
                                    'time_ms': (global_peak_idx / sampling_rate) * 1000,
                                    'amplitude': event_amplitude,
                                    'local_baseline': local_baseline,
                                    'peak_value': event_peak_value
                                })
                                event_local_baselines.append(local_baseline)

            all_events.sort(key=lambda x: x['index'])
            unique_events = []
            min_distance_samples = int(sampling_rate * 5 * 0.001)  # 5ms minimum

            for event in all_events:
                if not unique_events or event['index'] - unique_events[-1]['index'] >= min_distance_samples:
                    unique_events.append(event)

            num_events = len(unique_events)
            event_times = [event['time_ms'] for event in unique_events]
            event_indices = [event['index'] for event in unique_events]
            event_amplitudes = [event['amplitude'] for event in unique_events]
            event_peak_values = [event['peak_value'] for event in unique_events]
            event_local_baselines = [event['local_baseline'] for event in unique_events]

            results[trace_index] = {
                'basic_stats': {
                    #'mean_vm': mean_vm,
                    #'median_vm': median_vm,
                    #'std_vm': std_vm,
                },
                'event_stats': {
                    'num_events': num_events,
                    'event_times': event_times,
                    'event_indices': event_indices,
                    'event_amplitudes': event_amplitudes.tolist() if hasattr(event_amplitudes, 'tolist') else list(event_amplitudes),
                    'event_local_baselines': event_local_baselines,
                    'event_peak_values': event_peak_values,
                    'median_amplitude': np.median(event_amplitudes) if len(event_amplitudes) > 0 else np.nan,
                }
            }

    results_to_group = prepare_results_for_grouping(results)
    result_dict = {file_index + 1: results_to_group}

    time_window_ms = 5.0
    for neuron in result_dict:
        grouped_results = group_events_by_time(result_dict[neuron], time_threshold=time_window_ms)
    
    st.header('Subthreshold Event Analysis Results')
    st.text('Below are the grouped subthreshold events, returned as lists of tuples where each tuple is (trace index, ' \
            'time of event (ms), local baseline (mV), peak value (mV), event amplitude from local baseline (mV)). ' \
            'Only events with amplitude >= 1.0 mV are considered. ' \
            'If no subthreshold events are detected in a trace, it is represented as (-1, -1, -1, -1). ' \
            'Each group of events is displayed on a new line. ' \
            'Trace indices are 0-based.')
    st.text('This is only the first step of the analysis, further steps involve manually verifying these events, as for ' \
            'few traces, multiple events in close proximity may be detected, and by manual verification with the traces '
            'a single event can be chosen based on either which event occured first or which event resulted in the '
            'largest peak value of the membrane potential. Once this decision is made, the final list of events for ' \
            'each neuron can be passed on to a dictionary for further analysis.')
    
    st.subheader(f'Subthreshold events detected for Neuron {file_index + 1}: ')

    for group in grouped_results:
        a = sorted(group, key=lambda x: x[1])
        
        if grouped_results.index(group) == len(grouped_results) - 1:
            st.text(f'{a}')
        else:
            st.text(f'{a},')
    
def prepare_results_for_grouping(results):
    """
    Prepare results for grouping by extracting relevant event information.
    Args:
        results: Dictionary of results from analyze_traces function
    Returns:
        List of lists, where each inner list contains tuples of
        (time_of_event, baseline, peak_value, event_amplitude) for each trace
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
            event_times = [round(t, 3) for t in event_times]
            event_amps = [round(t, 3) for t in event_amps]
            event_peak_vals = [round(t, 3) for t in event_peak_vals]
            event_local_base = [round(t, 3) for t in event_local_base]


            for i in range(len(event_times)):
                if event_amps[i] >= 1.0:
                    trace_event_list.append((event_times[i], event_local_base[i], event_peak_vals[i], event_amps[i]))
                    
        else:
            trace_event_list.append((-1, -1, -1, -1))

            
        neuron_results.append(trace_event_list)
    return neuron_results

def group_events_by_time(traces, time_threshold=5.0):
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
            event_time, baseline_value, peak_value, event_amp = sub_event
            all_events.append((trace_id, event_time, baseline_value, peak_value, event_amp))

    all_events.sort(key=lambda x: x[1]) 
    
    grouped_events = []
    current_group = []
    
    for sub_event in all_events:
        if not current_group:

            current_group.append(sub_event)
        else:
            
            dyn_mean = np.mean([e[1] for e in current_group])

            time_diff = abs(sub_event[1] - dyn_mean) # dynamic mean time, so the events are 
            #time_diff = abs(sub_event[1] - current_group[0][1])
            if time_diff <= time_threshold and time_diff >= -time_threshold:
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
        
        st.subheader("Trace Selection & Plotting")
        
        col1, col2 = st.columns(2)
        
        with col1:

            select_all = st.checkbox("Select All Traces (Would be ideal if there are less than 10 traces)", value=False)
            if select_all:
                selected_traces = list(range(num_traces))
            else:
                selected_traces = st.multiselect(
                    "Select Traces to Plot",
                    options=range(num_traces),
                    format_func=lambda x: f"Trace {x + 1}",
                    default=[0] if num_traces > 0 else []
                )
        
        with col2:
            plot_type = st.radio("Plot Type", options=['overlay', 'subplots'])
        
        sampling_rate = 40000 #Hz
        
        select_plot_preprocessing = st.selectbox('Smoothen trace? (uses the Savgol filter from scipy.signal)', ['No', 'Yes'])
        if select_plot_preprocessing == 'Yes':
            smooth_window_ms = st.number_input('Smoothing Window (ms)', min_value=0.5, max_value=5.0, value=1.0, step=0.5)
            polyorder = st.number_input('Polynomial Order for Smoothing', min_value=1, max_value=10, value=2, step=1)  
        else:
            smooth_window_ms = None
            polyorder = None

        if select_plot_preprocessing == 'No':
            preprocess = False
            plot_traces(viewer, selected_traces, preprocess, smooth_window_ms, polyorder, plot_type, sampling_rate)
        elif select_plot_preprocessing == 'Yes':
            preprocess = True
            plot_traces(viewer, selected_traces, preprocess, smooth_window_ms, polyorder, plot_type, sampling_rate)
        
        selected_traces_for_analysis = list(range(num_traces))
        if select_plot_preprocessing == 'No':
            preprocess = False
            analyze_traces(viewer, viewer.current_file_index, selected_traces_for_analysis, preprocess, smooth_window_ms, polyorder, sampling_rate)
        elif select_plot_preprocessing == 'Yes':
            preprocess = True
            analyze_traces(viewer, viewer.current_file_index, selected_traces_for_analysis, preprocess, smooth_window_ms, polyorder, sampling_rate)
        
if __name__ == "__main__":
    main()