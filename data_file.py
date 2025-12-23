import re
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
        self.antenna_frequencies_hz : list[float] = []
        self.info : dict[str, str] = {}


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


    def trace_cumulative_distances_meters(self) -> np.ndarray|None:
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
                    # Store both start and end frequencies
                    self.antenna_frequencies_hz = [float(sweep_freq_start), float(sweep_freq_end)]
                elif sweep_freq_start > 0:
                    self.antenna_frequencies_hz = [float(sweep_freq_start)]
                elif sweep_freq_end > 0:
                    self.antenna_frequencies_hz = [float(sweep_freq_end)]

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
                    self.trace_offset_meters[i] = trace_header[segyio.TraceField.offset] * factor_length_2_mks

                    x, y = trace_header[segyio.TraceField.CDP_X], trace_header[segyio.TraceField.CDP_Y]
                    if not isinstance(x, float) or not isinstance(y, float):
                        x = float(x)
                        y = float(y)
                    # Skip invalid coordinates
                    if (x == 0 and y == 0) or not np.isfinite(x) or not np.isfinite(y):
                        continue
                    count_valid_coords += 1
                    scalar = trace_header[segyio.TraceField.SourceGroupScalar]
                    if scalar < 0:
                        scalar = 1/abs(float(scalar))
                    elif scalar > 0:
                        scalar = float(scalar)
                    else:
                        scalar = 1.0
                    x *= float(scalar)
                    y *= float(scalar)
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
        # Create SEGY file with data and metadata
        try:
            spec = segyio.spec()
            spec.samples = range(self.data.shape[0])
            spec.tracecount = self.data.shape[1]
            spec.format = 1  # 4-byte IBM float

            with segyio.create(self.filename, spec) as f:
                # Write binary header metadata
                f.bin[segyio.BinField.Interval] = int(self.time_interval_seconds * 1_000_000)  # Convert to microseconds
                f.bin[segyio.BinField.MeasurementSystem] = 1  # MKS (all data stored in meters)
                if self.antenna_frequencies_hz:
                    f.bin[segyio.BinField.SweepFrequencyStart] = int(self.antenna_frequencies_hz[0])
                    if len(self.antenna_frequencies_hz) > 1:
                        f.bin[segyio.BinField.SweepFrequencyEnd] = int(self.antenna_frequencies_hz[-1])

                # Write trace data and headers
                for i, trace in enumerate(self.data.T):
                    f.trace[i] = trace
                    # Write trace header metadata
                    if self.trace_time_delays_seconds is not None and i < len(self.trace_time_delays_seconds):
                        f.header[i][segyio.TraceField.DelayRecordingTime] = int(self.trace_time_delays_seconds[i] * 1000)  # Convert to ms
                    if self.trace_offset_meters is not None and i < len(self.trace_offset_meters):
                        f.header[i][segyio.TraceField.offset] = int(self.trace_offset_meters[i])
                    if self.trace_coords_meters is not None and i < len(self.trace_coords_meters):
                        # Use scalar of -100 for centimeter precision
                        f.header[i][segyio.TraceField.SourceGroupScalar] = -100
                        f.header[i][segyio.TraceField.CDP_X] = int(self.trace_coords_meters[i, 0] * 100)
                        f.header[i][segyio.TraceField.CDP_Y] = int(self.trace_coords_meters[i, 1] * 100)
                        f.header[i][segyio.TraceField.CoordinateUnits] = 1  # length (meters)

        except Exception as e:
            self.error = f"Error saving SEGY file: {str(e)}"
            return False
        return True


    
    ################### MALA file functions ###################
    
    def _load_mala_data(self) -> bool:
        try:
            file_base, _ = os.path.splitext(self.filename)
            data, _ = readMALA(file_base)
            self.data = np.array(data)
            if self.data is None or self.data.size == 0:
                self.error = "No data found in MALA file"
                return False
            nt, nx = self.data.shape

            # Extract metadata from MALA header
            self.info = {}
            with open(file_base + '.rad', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1) # split on first colon
                        self.info[key.strip()] = value.strip()
            # Calculate sample interval (dt) from TIMEWINDOW and SAMPLES
            # TIMEWINDOW is in nanoseconds
            timewindow_ns = float(self.info.get('TIMEWINDOW', 0))
            if timewindow_ns > 0 and nt > 1:
                dt_ns = timewindow_ns / (nt - 1)
                self.time_interval_seconds = dt_ns / 1_000_000_000.0  # Convert nanoseconds to seconds
            else:
                self.time_interval_seconds = 1.0

            # Extract signal position (time zero offset) from MALA header
            # SIGNAL POSITION is in nanoseconds
            signal_position_ns = float(self.info.get('SIGNAL POSITION', 0))
            self.time_delay_seconds = signal_position_ns / 1_000_000_000.0  # Convert nanoseconds to seconds

            # Extract antenna frequency from MALA header (in MHz, convert to Hz)
            antenna_type = self.info.get('ANTENNAS', None)
            if antenna_type is None:
                self.error = "Antenna type is not set in MALA file"
                return False
            antenna_type = antenna_type.lower()
            if 'ghz' in antenna_type:
                antenna_ghz = antenna_type.split('ghz')[0].strip()
                antenna_ghz = float(antenna_ghz)
                self.antenna_frequencies_hz = [antenna_ghz * 1_000_000_000.0]
            elif 'mhz' in antenna_type:
                antenna_mhz = antenna_type.split('mhz')[0].strip()
                antenna_mhz = float(antenna_mhz)
                self.antenna_frequencies_hz = [antenna_mhz * 1_000_000.0]
            elif 'khz' in antenna_type:
                antenna_kHz = antenna_type.split('khz')[0].strip()
                antenna_kHz = float(antenna_kHz)
                self.antenna_frequencies_hz = [antenna_kHz * 1_000.0]
            elif 'hz' in antenna_type:
                antenna_Hz = antenna_type.split('hz')[0].strip()
                antenna_Hz = float(antenna_Hz)
                self.antenna_frequencies_hz = [antenna_Hz]
            else:
                # replace all non-numeric characters with spaces (keep digits, dots, e, +, -)
                antenna_frequency_str = re.sub(r'[^0-9.e+-]', ' ', antenna_type)
                frequency_candidates = [float(f) for f in antenna_frequency_str.split() if f.strip()]
                if not frequency_candidates:
                    self.error = f"Could not parse antenna frequency from: {antenna_type}"
                    return False
                antenna_frequency_mhz = max(frequency_candidates)
                self.antenna_frequencies_hz = [antenna_frequency_mhz * 1_000_000.0]

            # Distance interval from MALA header
            distance_interval = float(self.info.get('DISTANCE INTERVAL', 0))
            if distance_interval <= 0:
                self.error = "Distance interval is not set in MALA file"
                return False
            # read distance interval units from MALA header and convert to MKS
            distance_interval_units = self.info.get('LENGTH UNITS', 'm')
            if distance_interval_units == 'm':
                file_unit_system = UnitSystem.MKS
            elif distance_interval_units == 'ft':
                file_unit_system = UnitSystem.IMPERIAL
            else:
                self.error = f"Unsupported distance interval units: {distance_interval_units}"
                return False
            factor_length_2_mks = UnitSystem.convert_length_factor(file_unit_system, UnitSystem.MKS)
            # Convert offset to meters
            self.offset_meters = float(self.info.get('OFFSET', 0)) * factor_length_2_mks
            # Create linear coordinates based on distance interval (in meters)
            x_coords = np.arange(nx) * distance_interval * factor_length_2_mks
            self.trace_coords_meters = np.column_stack([x_coords, np.zeros_like(x_coords)])
        except Exception as e:
            self.error = f"Error loading MALA file: {str(e)}"
            return False
        return True
        



    def _save_mala_data(self) -> bool:
        try:
            # Save the data file
            if self.file_extension == 'rd3':
                data_normalized = self.data / np.max(np.abs(self.data))
                data_int16 = (data_normalized * 32767).astype(np.int16)
                data_int16.T.tofile(self.filename)
            else:
                self.data.T.tofile(self.filename)

            # Save the .rad header file
            file_base, _ = os.path.splitext(self.filename)
            rad_filename = file_base + '.rad'

            # Update self.info with current values (all stored in MKS)
            nt, nx = self.data.shape
            self.info['SAMPLES'] = str(nt)
            # TIMEWINDOW in nanoseconds
            timewindow_ns = self.time_interval_seconds * (nt - 1) * 1_000_000_000.0
            self.info['TIMEWINDOW'] = f"{timewindow_ns:.6f}"
            # SIGNAL POSITION in nanoseconds
            self.info['SIGNAL POSITION'] = f"{self.time_delay_seconds * 1_000_000_000.0:.6f}"
            # DISTANCE INTERVAL in meters
            if self.trace_coords_meters is not None and nx > 1:
                distance_interval = np.median(np.diff(self.trace_coords_meters[:, 0]))
                self.info['DISTANCE INTERVAL'] = f"{distance_interval:.10f}"
            self.info['LENGTH UNITS'] = 'm'
            self.info['OFFSET'] = f"{self.offset_meters:.6f}"
            self.info['LAST TRACE'] = str(nx)
            # Antenna frequency in MHz
            if self.antenna_frequencies_hz:
                antenna_mhz = self.antenna_frequencies_hz[0] / 1_000_000.0
                if antenna_mhz >= 1000:
                    self.info['ANTENNAS'] = f"{antenna_mhz / 1000:.1f} GHz"
                else:
                    self.info['ANTENNAS'] = f"{antenna_mhz:.0f} MHz"

            # Write the .rad file
            with open(rad_filename, 'w') as f:
                for key, value in self.info.items():
                    f.write(f"{key}:{value}\n")

        except Exception as e:
            self.error = f"Error saving MALA file: {str(e)}"
            return False
        return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    df = DataFile(filename)

    if not df.load():
        print(f"Error: {df.error}")
        sys.exit(1)

    print(f"File: {df.filename}")
    print(f"Format: {df.file_extension.upper()}")
    print()
    print("=== Data Statistics ===")
    print(f"Data shape: {df.data.shape} (samples x traces)")
    print(f"Data type: {df.data.dtype}")
    print(f"Data range: [{df.data.min():.6g}, {df.data.max():.6g}]")
    print()
    print("=== Time ===")
    print(f"Time interval: {df.time_interval_seconds * 1e9:.3f} ns")
    print(f"Time delay: {df.time_delay_seconds * 1e9:.3f} ns")
    print(f"Time window: {df.time_interval_seconds * (df.data.shape[0] - 1) * 1e9:.3f} ns")
    print()
    print("=== Spatial ===")
    print(f"Offset: {df.offset_meters:.6f} m")
    if df.trace_coords_meters is not None:
        dist = df.trace_cumulative_distances_meters()
        if dist is not None:
            print(f"Distance range: [{dist[0]:.3f}, {dist[-1]:.3f}] m")
            if len(dist) > 1:
                intervals = np.diff(dist)
                print(f"Trace spacing: {np.median(intervals):.6f} m (median)")
    print()
    print("=== Antenna ===")
    if df.antenna_frequencies_hz:
        freqs = [f/1e6 for f in df.antenna_frequencies_hz]
        print(f"Antenna frequency: {freqs} MHz")
    else:
        print("Antenna frequency: not available")
    print()
    if df.info:
        print("=== Header Info ===")
        for key, value in df.info.items():
            print(f"  {key}: {value}")
