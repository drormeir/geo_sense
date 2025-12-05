# Global Display Settings

## Overview

Global Display Settings control application-wide display parameters that affect all seismic subwindows. These settings are accessible from the Settings menu and persist across sessions.

---

## Accessing Global Settings

### From the Menu Bar
```
Settings → Global Settings...
```

The Global Settings dialog will appear as a modeless window, allowing you to adjust settings while viewing their effects on open seismic data windows in real-time.

---

## Settings Categories

### Plot Margins (pixels)

Control the spacing around seismic data plots. All margin values are specified in pixels.

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Base vertical** | Basic top/bottom spacing around the plot | 20 px | 0-200 px |
| **Horizontal axes** | Extra space allocated for horizontal axis labels (top/bottom) | 30 px | 0-200 px |
| **Base horizontal** | Basic left/right spacing around the plot | 10 px | 0-200 px |
| **Vertical axes** | Extra space allocated for vertical axis labels (left/right) | 70 px | 0-200 px |
| **Colorbar width** | Width of the colorbar when displayed | 30 px | 0-200 px |

#### How Margins Work

Margins are calculated dynamically based on which display elements are visible:

**Top Margin:**
- Base vertical margin (always applied)
- + Horizontal axes margin (if top border labels are enabled)
- + Additional base vertical margin (if file name in plot is enabled)

**Bottom Margin:**
- Base vertical margin (always applied)
- + Horizontal axes margin (if bottom border labels are enabled)

**Left Margin:**
- Base horizontal margin (always applied)
- + Vertical axes margin (if left border labels are enabled)

**Right Margin:**
- Base horizontal margin (always applied)
- + Vertical axes margin (if right border labels are enabled)
- + Colorbar width + margins (if colorbar is visible)

---

### Unit System

Controls the unit system used throughout the application for distance and depth measurements.

| Option | Description | Distance Units | Depth Units |
|--------|-------------|----------------|-------------|
| **MKS** | Meter-Kilogram-Second (SI units) | meters (m), kilometers (km) | meters (m) |
| **Imperial** | Foot-Pound-Second | feet (ft), miles (mi) | feet (ft) |

**Note:** The unit system setting is currently defined but unit conversion features are planned for future implementation.

---

## Live Update Behavior

The Global Settings dialog provides **live updates**:

- Changes take effect **immediately** as you adjust settings
- All open seismic subwindows update their layouts in real-time
- You can see the visual effects before closing the dialog
- Click **OK** to keep changes, **Cancel** to revert to previous values

---

## Dialog Mode

The Global Settings dialog is **modeless**:

- You can interact with seismic data windows while the dialog is open
- Adjust settings and observe the effects on your data simultaneously
- The dialog stays open until you explicitly close it
- Only one Global Settings dialog can be open at a time

---

## Session Persistence

Global settings are automatically saved and restored:

- Settings are saved when you exit the application (if session saving is enabled)
- Settings are restored when you start the application (if session loading is enabled)
- Settings persist in the session file: `~/.config/uas_sessions/default.json`

### Controlling Session Persistence

Session behavior can be controlled via command-line arguments:

```bash
# Normal mode (default): load and save settings
python seismic_app.py

# No session read/write: starts with default settings
python seismic_app.py --session-mode -1

# Fresh start but save on exit: use defaults, save changes
python seismic_app.py --session-mode 0
```

See `COMMAND_LINE_USAGE.md` for more details on session modes.

---

## Use Cases

### Adjusting for High-DPI Displays

If labels appear too cramped on high-resolution displays:

1. Open Settings → Global Settings
2. Increase **Vertical axes** margin (e.g., 80-100 px)
3. Increase **Horizontal axes** margin (e.g., 40-50 px)
4. Observe the changes in real-time
5. Click **OK** when satisfied

### Maximizing Data Display Area

To dedicate more screen space to seismic data:

1. Open Settings → Global Settings
2. Reduce **Base vertical** margin (e.g., 10 px)
3. Reduce **Base horizontal** margin (e.g., 5 px)
4. Reduce **Colorbar width** if using colorbars (e.g., 20 px)
5. Click **OK** to apply

### Creating Presentation-Ready Plots

For cleaner plots suitable for presentations:

1. Open Settings → Global Settings
2. Increase **Base vertical** margin (e.g., 30 px) for breathing room
3. Increase **Base horizontal** margin (e.g., 20 px)
4. Ensure adequate **Colorbar width** (e.g., 40 px) for readability
5. Click **OK** to apply

---

## Technical Details

### Observer Pattern

Global Settings uses the observer pattern to notify all seismic subwindows when settings change:

- Each subwindow registers as a listener when created
- When settings change, all listeners are notified automatically
- Subwindows update their layouts to reflect new margin values
- Listeners are automatically unregistered when subwindows close

### Margin Calculation

Margins are calculated in pixels and converted to figure-relative coordinates:

```
proportion = margin_px / (figure_dimension_px)
```

This ensures consistent pixel-based spacing regardless of window size or DPI.

### Singleton Pattern

GlobalSettings is implemented as a singleton:

- Only one instance exists throughout the application lifetime
- All windows access the same settings data
- Changes are immediately visible to all components
- Thread-safe access to settings (UI runs on main thread)

---

## Troubleshooting

### Settings Don't Persist Across Sessions

**Cause:** Session saving may be disabled.

**Solution:** Ensure you're running in normal mode (session-mode 1):
```bash
python seismic_app.py  # Default is mode 1
```

### Labels Are Cut Off

**Cause:** Insufficient margin space for axis labels.

**Solution:** Increase the relevant margin:
- For left/right labels: increase **Vertical axes** margin
- For top/bottom labels: increase **Horizontal axes** margin

### Too Much White Space

**Cause:** Margins are set too large.

**Solution:** Reduce margin values in Global Settings dialog.

### Changes Don't Apply

**Cause:** May be a rendering issue.

**Solution:** Try these steps:
1. Close and reopen the seismic subwindow
2. Restart the application
3. Check that settings were saved (use `--print-session` flag)

---

## Keyboard Shortcuts

Currently, there are no keyboard shortcuts for Global Settings. Access is through:
- Menu: **Settings → Global Settings...**

---

## Related Documentation

- `COMMAND_LINE_USAGE.md` - Session modes and command-line options
- `DISPLAY_SETTINGS.md` - Per-window display settings (axes, colormaps, etc.)

---

## Future Enhancements

Planned features for Global Settings:

- Font size control for axis labels and titles
- Color scheme presets (light/dark themes)
- Full unit conversion implementation for Imperial units
- Export/import settings profiles
- Per-monitor DPI scaling settings
