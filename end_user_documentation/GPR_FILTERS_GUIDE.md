# GPR Filters Guide

This document describes the common frequency and signal processing filters used in Ground Penetrating Radar (GPR) data processing, organized by category.

---

## Filter Categories Overview

| Category | Purpose | When to Apply |
|----------|---------|---------------|
| **Preprocessing** | Remove DC offset, baseline drift | First step, before other filters |
| **Frequency Domain** | Remove unwanted frequency bands | After preprocessing |
| **Amplitude** | Normalize signal strength | After frequency filtering |
| **Spatial** | Remove coherent noise across traces | Depends on noise type |

---

## 1. Preprocessing Filters

These filters prepare the data for subsequent processing by removing baseline artifacts.

### DC Removal

| Property | Description |
|----------|-------------|
| **Purpose** | Remove zero-frequency (DC) offset from traces |
| **How it works** | Subtracts the mean value from each trace |
| **When to use** | Always - fundamental first step |
| **Parameters** | Method: trace (per-trace), global (all data), sliding (running mean) |
| **Effect** | Centers waveforms around zero amplitude |

**Methods comparison:**
| Method | Description | Best for |
|--------|-------------|----------|
| Trace | Removes mean of each trace independently | Standard GPR data |
| Global | Removes overall mean of entire dataset | Uniform DC offset |
| Sliding | Removes running mean (acts like highpass) | Slowly varying drift |

### Dewow

| Property | Description |
|----------|-------------|
| **Purpose** | Remove low-frequency "wow" artifact from GPR data |
| **How it works** | Subtracts a running mean/median from each trace |
| **When to use** | After DC removal, when low-frequency oscillation is present |
| **Parameters** | Window length (ns), Method (mean or median) |
| **Effect** | Removes exponentially decaying artifact near time zero |

**What causes wow:**
- Direct coupling between transmitter and receiver antennas
- Instrument DC drift
- Near-field electromagnetic effects

**Window length guidelines:**
| Antenna Frequency | Suggested Window |
|-------------------|------------------|
| 100-200 MHz | 20-50 ns |
| 400-500 MHz | 10-20 ns |
| 800-1600 MHz | 5-10 ns |

---

## 2. Frequency Domain Filters

These filters operate on the frequency content of the signal to remove unwanted frequency bands.

### Filter Types Comparison

| Filter | Passband | Stopband | Phase | Complexity | Best for |
|--------|----------|----------|-------|------------|----------|
| **Lowpass** | f < fc | f > fc | Depends on design | Low | Remove high-frequency noise |
| **Highpass** | f > fc | f < fc | Depends on design | Low | Remove low-frequency drift |
| **Bandpass** | f1 < f < f2 | f < f1 or f > f2 | Depends on design | Medium | Standard GPR processing |
| **Bandstop** | f < f1 or f > f2 | f1 < f < f2 | Depends on design | Medium | Remove specific interference |

### FIR (Finite Impulse Response) Filters

| Property | Description |
|----------|-------------|
| **Purpose** | Frequency filtering with guaranteed stability and linear phase |
| **How it works** | Convolution with finite-length filter coefficients |
| **Advantages** | Always stable, linear phase (no waveform distortion) |
| **Disadvantages** | Requires more coefficients for sharp cutoff |
| **Parameters** | Cutoff frequencies, number of taps, design method, window type |

**FIR Design Methods:**
| Design | Description | Characteristics |
|--------|-------------|-----------------|
| Window | Uses window function (Hamming, Hann, etc.) | Simple, good general purpose |
| Least-squares | Minimizes squared error | Smooth response |
| Parks-McClellan | Equiripple (Remez algorithm) | Optimal minimax, sharpest rolloff |

**Number of Taps Guidelines:**
- More taps = sharper transition band
- Rule of thumb: `N ≈ 4 / (transition_width / sampling_rate)`
- Typical range: 51-201 taps

### IIR (Infinite Impulse Response) Filters

| Property | Description |
|----------|-------------|
| **Purpose** | Efficient frequency filtering with sharp cutoff |
| **How it works** | Recursive filter with feedback |
| **Advantages** | Sharp cutoff with fewer coefficients |
| **Disadvantages** | Can be unstable, nonlinear phase (use zero-phase filtering) |
| **Parameters** | Cutoff frequencies, order, design type |

**IIR Design Types:**
| Design | Passband | Stopband | Phase | Best for |
|--------|----------|----------|-------|----------|
| **Butterworth** | Maximally flat | Smooth rolloff | Nonlinear | General purpose, minimal distortion |
| **Bessel** | Near-linear phase | Gradual rolloff | Most linear | Preserving waveform shape |
| **Chebyshev I** | Ripple allowed | Steep rolloff | Nonlinear | Sharp cutoff needed |
| **Chebyshev II** | Flat | Ripple allowed | Nonlinear | Flat passband required |
| **Elliptic** | Ripple allowed | Ripple allowed | Most nonlinear | Steepest possible rolloff |

**Order Guidelines:**
- Higher order = steeper rolloff but more ringing
- Typical range: 2-6 for GPR
- Use zero-phase filtering (filtfilt) to eliminate phase distortion

### Ormsby Bandpass Filter

| Property | Description |
|----------|-------------|
| **Purpose** | Trapezoidal bandpass filter with smooth transitions |
| **How it works** | Frequency-domain multiplication with trapezoidal response |
| **Advantages** | Intuitive 4-corner frequency specification, minimal ringing |
| **Disadvantages** | Less flexible than FIR/IIR |
| **Parameters** | F1 (low cut), F2 (low pass), F3 (high pass), F4 (high cut), taper |

**Frequency Response Shape:**
```
     Response
        1 |      ___________
          |     /           \
          |    /             \
        0 |___/               \___
          F1  F2           F3  F4  Frequency
```

**Taper Types:**
| Taper | Description | Characteristic |
|-------|-------------|----------------|
| Linear | Straight line | Simple, some ringing |
| Cos² (Hann) | Cosine squared | Smooth, low ringing |
| Hamming | Modified cosine | Slightly less smooth than Hann |
| Blackman | Triple cosine | Smoothest, widest transition |

**Frequency Selection Guidelines (based on antenna frequency):**
| Antenna | F1 | F2 | F3 | F4 |
|---------|----|----|----|----|
| 100 MHz | 20 | 50 | 150 | 180 MHz |
| 250 MHz | 50 | 125 | 375 | 450 MHz |
| 500 MHz | 100 | 250 | 750 | 900 MHz |
| 800 MHz | 160 | 400 | 1200 | 1440 MHz |

*Rule of thumb: F1 ≈ 0.2×Fc, F2 ≈ 0.5×Fc, F3 ≈ 1.5×Fc, F4 ≈ 1.8×Fc (where Fc = antenna center frequency)*

---

## 3. Amplitude Filters

These filters modify signal amplitude to improve visualization and interpretation.

### AGC (Automatic Gain Control)

| Property | Description |
|----------|-------------|
| **Purpose** | Normalize amplitude over time to enhance weak signals |
| **How it works** | Divides signal by local RMS amplitude |
| **When to use** | When deep reflections are too weak to see |
| **Parameters** | Window length (ms), Target RMS |
| **Effect** | Equalizes amplitude, loses true amplitude information |

**Window Length Guidelines:**
| Data Type | Suggested Window |
|-----------|------------------|
| Shallow GPR (< 2m) | 50-200 ms |
| Medium GPR (2-10m) | 200-500 ms |
| Deep GPR (> 10m) | 500-1000 ms |

**Caution:** AGC destroys true amplitude relationships. Do not use AGC before:
- Amplitude analysis
- AVO (Amplitude vs Offset) analysis
- Quantitative interpretation

### Other Amplitude Filters (Not Yet Implemented)

| Filter | Purpose | Description |
|--------|---------|-------------|
| **Gain** | Time-varying amplitude correction | Multiply by t^n or exponential function |
| **SEC (Spherical & Exponential Compensation)** | Correct for geometric spreading | Compensates for 1/r² amplitude decay |
| **Trace Normalization** | Equalize trace amplitudes | Scale each trace to same max/RMS |
| **Clip** | Limit extreme amplitudes | Prevents display saturation |

---

## 4. Spatial Filters

These filters operate across traces to remove spatially coherent noise.

### Background Removal (BGR)

| Property | Description |
|----------|-------------|
| **Purpose** | Remove horizontal banding (features constant across traces) |
| **How it works** | Subtracts average/median trace from all traces |
| **When to use** | When horizontal bands obscure dipping reflectors |
| **Parameters** | Method (mean/median), number of traces, trace selection |
| **Effect** | Removes ringing, direct wave; preserves hyperbolas, dipping layers |

**What BGR Removes:**
- Antenna ringing (horizontal bands)
- System noise and DC offset
- Direct wave and ground wave
- Any feature that doesn't vary laterally

**What BGR Preserves:**
- Point reflectors (hyperbolas)
- Dipping layers
- Lateral variations in stratigraphy

**Method Comparison:**
| Method | Description | Best for |
|--------|-------------|----------|
| Mean | Average of selected traces | Normal data |
| Median | Median of selected traces | Data with outliers/spikes |

### Other Spatial Filters (Not Yet Implemented)

| Filter | Purpose | Description |
|--------|---------|-------------|
| **F-K Filter** | Remove dipping noise | Filter in frequency-wavenumber domain |
| **Migration** | Collapse diffractions | Move energy to true subsurface position |
| **Stacking** | Reduce random noise | Average multiple traces |
| **Spatial Smoothing** | Reduce trace-to-trace noise | Running average across traces |

---

## Recommended Processing Flow

A typical GPR processing sequence:

```
1. DC Removal          → Center data around zero
        ↓
2. Dewow               → Remove low-frequency artifact
        ↓
3. Time Zero Correction → Align first arrivals (manual/auto)
        ↓
4. Background Removal  → Remove horizontal banding
        ↓
5. Bandpass Filter     → Remove out-of-band noise
        ↓
6. AGC (optional)      → Enhance deep reflections for display
```

**Notes:**
- Order matters! Each step assumes previous steps are complete.
- Not all steps are always necessary - depends on data quality.
- AGC should be last and only for display, not analysis.

---

## Filter Selection Decision Tree

```
Is there a DC offset or baseline drift?
    YES → Apply DC Removal

Is there low-frequency "wow" near time zero?
    YES → Apply Dewow

Are there horizontal bands (ringing)?
    YES → Apply Background Removal

Is there high-frequency noise?
    YES → Apply Lowpass or Bandpass filter

Is there low-frequency drift remaining?
    YES → Apply Highpass or Bandpass filter

Are deep reflections too weak to see?
    YES → Apply AGC (for display only)
```

---

## Quick Reference Table

| Filter | Category | Removes | Preserves | Key Parameter |
|--------|----------|---------|-----------|---------------|
| DC Removal | Preprocessing | DC offset | All AC content | Method |
| Dewow | Preprocessing | Low-freq wow | Reflections | Window (ns) |
| FIR | Frequency | Out-of-band frequencies | In-band frequencies | Cutoff, taps |
| IIR | Frequency | Out-of-band frequencies | In-band frequencies | Cutoff, order |
| Ormsby | Frequency | Out-of-band frequencies | In-band frequencies | F1, F2, F3, F4 |
| AGC | Amplitude | Amplitude variations | Waveform shape | Window (ms) |
| BGR | Spatial | Horizontal features | Dipping/point features | Method, traces |

---

## Glossary

| Term | Definition |
|------|------------|
| **Bandpass** | Filter that passes frequencies within a range |
| **Cutoff frequency** | Frequency at which filter response is -3dB |
| **DC** | Direct Current; zero-frequency component |
| **FIR** | Finite Impulse Response - filter with finite memory |
| **IIR** | Infinite Impulse Response - filter with feedback |
| **Linear phase** | All frequencies delayed equally (no waveform distortion) |
| **Nyquist frequency** | Half the sampling rate; maximum representable frequency |
| **Passband** | Frequency range that passes through filter |
| **Ripple** | Amplitude variations in passband or stopband |
| **Rolloff** | Rate of attenuation in transition band |
| **RMS** | Root Mean Square - measure of signal amplitude |
| **Stopband** | Frequency range that is attenuated by filter |
| **Taper** | Gradual transition function |
| **Zero-phase** | Filtering forward and backward to eliminate phase shift |
| **Wow** | Low-frequency artifact in GPR from antenna coupling |
