from gprpy.toolbox.gprIO_MALA import readMALA
import numpy as np
import os
import segyio
from global_settings import UnitSystem
from uas import interpolate_inplace_nan_values

class DataFile:
    segy_coord_unit_map = {
        1: "length", # meters or feet
        2: "arcsec", # seconds of arc
        3: "deg", # decimal degrees
        4: "DMS", # degrees, minutes, seconds
    }
    segy_measurement_system_map = {
        1: UnitSystem.MKS,
        2: UnitSystem.IMPERIAL,
    }

    def __init__(self, filename: str):
        self.filename = filename
        self.file_extension = os.path.splitext(filename)[1].lower()[1:]
        self.data : np.ndarray | None = None
        self.error : str = ""
        self.time_interval_seconds : float = 0.0
        self.trace_time_delays_seconds : np.ndarray | None = None
        self.trace_offset_meters : np.ndarray | None = None
        self.trace_coords_meters : np.ndarray | None = None
        self.offset_meters : float = 0.0
        self.time_delay_seconds : float = 0.0
        self.antenna_frequency_hz : float = 0.0


    def load(self) -> bool:
        if not self.filename or not os.path.exists(self.filename) or not os.path.isfile(self.filename):
            self.error = f"Invalid file: {self.filename}"
            return False
        self.error = ""
        if self.file_extension in ['sgy', 'segy']:
            if not self._load_segy_data():
                return False
        elif self.file_extension in ['rd3', 'rd7']:
            if not self._load_mala_data():
                return False
        else:
            self.error = "Unsupported file extension"
            return False
        if self.data is not None:
            self.data = self.data.astype(np.float32)
        if self.trace_time_delays_seconds is not None:
            self.trace_time_delays_seconds = self.trace_time_delays_seconds.astype(np.float32)
        if self.trace_offset_meters is not None:
            self.trace_offset_meters = self.trace_offset_meters.astype(np.float32)
        if self.trace_coords_meters is not None:
            self.trace_coords_meters = self.trace_coords_meters.astype(np.float32)
        return True


    def save(self) -> bool:
        if self.data is None or self.data.size == 0:
            self.error = "No data to save"
            return False
        self.error = ""
        self.data = self.data.astype(np.float32)
        if self.trace_time_delays_seconds is not None:
            self.trace_time_delays_seconds = self.trace_time_delays_seconds.astype(np.float32)
        if self.trace_offset_meters is not None:
            self.trace_offset_meters = self.trace_offset_meters.astype(np.float32)
        if self.trace_coords_meters is not None:
            self.trace_coords_meters = self.trace_coords_meters.astype(np.float32)
        if self.file_extension in ['sgy', 'segy']:
            return self._save_segy_data()
        if self.file_extension in ['rd3', 'rd7']:
            return self._save_mala_data()
        self.error = "Unsupported file extension"
        return False


    def trace_comulative_distances_meters(self) -> np.ndarray|None:
        # Calculate cumulative distances along the survey line
        # Distance from trace i-1 to trace i for each trace
        if self.trace_coords_meters is None or self.trace_coords_meters.size == 0:
            return None
        try:
            trace_distances_meters = np.sqrt(np.sum(np.diff(self.trace_coords_meters, axis=0)**2, axis=1))
            # Cumulative sum with 0.0 prepended for first trace
            return np.cumsum(np.concatenate([[0.0], trace_distances_meters]))
        except Exception as e:
            error_message = f"Error calculating trace cumulative distances: {str(e)}"
            raise Exception(error_message)


    ################### SEGY file functions ###################
    def _load_segy_data(self) -> bool:
        try:
            with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                self.data = f.trace.raw[:].T
                if self.data is None or self.data.size == 0:
                    self.error = "No data found in SEGY file"
                    return False
                # Extract metadata from SEGY file
                # Sample interval from binary header (bytes 3217-3218, in microseconds per SEG-Y standard)
                dt_us = float(f.bin[segyio.BinField.Interval])
                if dt_us <= 0:
                    self.error = "Sample interval is not set in SEGY file"
                    return False
                self.time_interval_seconds = dt_us / 1_000_000.0  # Convert microseconds to seconds

                # Try to extract frequency from sweep frequency fields (sometimes used for antenna freq)
                sweep_freq_start = f.bin[segyio.BinField.SweepFrequencyStart]
                sweep_freq_end = f.bin[segyio.BinField.SweepFrequencyEnd]
                if sweep_freq_start > 0 and sweep_freq_end > 0:
                    # Use center frequency if both are set
                    self.antenna_frequency_hz = (sweep_freq_start + sweep_freq_end) / 2.0
                elif sweep_freq_start > 0:
                    self.antenna_frequency_hz = float(sweep_freq_start)
                elif sweep_freq_end > 0:
                    self.antenna_frequency_hz = float(sweep_freq_end)

                # Try to extract trace coordinates from trace headers
                num_traces = len(f.trace)
                if num_traces != self.data.shape[1]:
                    self.error = "Number of traces in SEGY file does not match number of traces in data"
                    return False
                self.trace_coords_meters = np.full((num_traces, 2), fill_value=np.nan)
                self.trace_offset_meters = np.full(num_traces, fill_value=np.nan)
                self.trace_time_delays_seconds = np.full(num_traces, fill_value=np.nan)

                # Read measurement system from binary header (bytes 3255-3256)
                measurement_system = f.bin[segyio.BinField.MeasurementSystem]
                file_unit_system = DataFile.segy_measurement_system_map.get(measurement_system, UnitSystem.MKS)
                factor_length_2_mks = UnitSystem.convert_length_factor(file_unit_system, UnitSystem.MKS)
                count_valid_coords = 0
                for i in range(num_traces):
                    trace_header = f.header[i]
                    # DelayRecordingTime from trace header (bytes 109-110, in milliseconds per SEG-Y standard)
                    delay_ms = trace_header[segyio.TraceField.DelayRecordingTime]
                    if delay_ms > 0:
                        self.trace_time_delays_seconds[i] = delay_ms / 1000.0  # Convert milliseconds to seconds
                    self.trace_offset_meters[i] = trace_header[segyio.TraceField.Offset] * factor_length_2_mks

                    x, y = trace_header[segyio.TraceField.CDP_X], trace_header[segyio.TraceField.CDP_Y]
                    if np.isnan(x) or np.isnan(y) or (x == 0 and y == 0):
                        continue
                    count_valid_coords += 1
                    scalar = trace_header[segyio.TraceField.SourceGroupScalar]
                    if scalar < 0:
                        x /= abs(scalar)
                        y /= abs(scalar)
                    elif scalar > 0:
                        x *= float(scalar)
                        y *= float(scalar)
                    else:
                        # ignore scalar or treat it as 1.0
                        pass
                    coord_units = trace_header[segyio.TraceField.CoordinateUnits]
                    if coord_units in DataFile.segy_coord_unit_map:
                        if DataFile.segy_coord_unit_map[coord_units] == "length":
                            x *= factor_length_2_mks
                            y *= factor_length_2_mks
                    self.trace_coords_meters[i] = [x, y]

                interpolate_inplace_nan_values(self.trace_time_delays_seconds)
                interpolate_inplace_nan_values(self.trace_offset_meters)
                self.offset_meters = np.median(self.trace_offset_meters)
                self.time_delay_seconds = np.median(self.trace_time_delays_seconds)
                if not count_valid_coords:
                    self.error = "No trace coordinates found in SEGY file"
                    return False
                interpolate_inplace_nan_values(self.trace_coords_meters)
        except Exception as e:
            self.error = f"Error loading SEGY file: {str(e)}"
            return False
        return True


    def _save_segy_data(self) -> bool:
        # Create a minimal SEGY file with current data
        try:
            spec = segyio.spec()
            spec.samples = range(self.data.shape[0])
            spec.tracecount = self.data.shape[1]
            spec.format = 1  # 4-byte IBM float

            with segyio.create(self.filename, spec) as f:
                for i, trace in enumerate(self.data.T):
                    f.trace[i] = trace

        except Exception as e:
            self.error = f"Error saving SEGY file: {str(e)}"
            return False
        return True


    
    ################### MALA file functions ###################
    
    def _load_mala_data(self) -> bool:
        try:
            file_base, _ = os.path.splitext(self.filename)
            data, info = readMALA(file_base)
            self.data = np.array(data)
            if self.data is None or self.data.size == 0:
                self.error = "No data found in MALA file"
                return False
            nt, nx = self.data.shape

            # Extract metadata from MALA header
            # Calculate sample interval (dt) from TIMEWINDOW and SAMPLES
            # TIMEWINDOW is in nanoseconds
            timewindow_ns = float(info.get('TIMEWINDOW', 0))
            if timewindow_ns > 0 and nt > 1:
                dt_ns = timewindow_ns / (nt - 1)
                self.time_interval_seconds = dt_ns / 1_000_000_000.0  # Convert nanoseconds to seconds
            else:
                self.time_interval_seconds = 1.0

            # Extract signal position (time zero offset) from MALA header
            # SIGNAL POSITION is in nanoseconds
            signal_position_ns = float(info.get('SIGNAL POSITION', 0))
            self.time_delay_seconds = signal_position_ns / 1_000_000_000.0  # Convert nanoseconds to seconds
            self.offset_meters = float(info.get('OFFSET', 0))

            # Extract antenna frequency from MALA header (in MHz, convert to Hz)
            antenna_mhz = float(info.get('ANTENNA', 0))
            self.antenna_frequency_hz = antenna_mhz * 1_000_000.0  # Convert MHz to Hz

            # Distance interval from MALA header
            distance_interval = float(info.get('DISTANCE INTERVAL', 0))
            if distance_interval <= 0:
                self.error = "Distance interval is not set in MALA file"
                return False
            # Create linear coordinates based on distance interval
            x_coords = np.arange(nx) * distance_interval
            self.trace_coords_meters = np.column_stack([x_coords, np.zeros_like(x_coords)])
        except Exception as e:
            self.error = f"Error loading MALA file: {str(e)}"
            return False
        return True
        



    def _save_mala_data(self) -> bool:
        try:
            if self.file_extension == 'rd3':
                data_normalized = self.data / np.max(np.abs(self.data))
                data_int16 = (data_normalized * 32767).astype(np.int16)
                data_int16.T.tofile(self.filename)
            else:
                self.data.T.tofile(self.filename)
        except Exception as e:
            self.error = f"Error saving MALA file: {str(e)}"
            return False
        return True


