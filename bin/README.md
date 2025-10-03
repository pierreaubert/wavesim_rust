# Room Acoustics Simulation Binary

Configurable room acoustics simulation with support for both 2D and 3D modes using the Helmholtz equation solver.

## Features

- **2D and 3D modes**: Choose between 2D (faster) or 3D (more realistic) simulations
- **Configurable dimensions**: Set room width, depth, and height (3D only)
- **Multiple sources**: Support for 1 or 2 acoustic sources
- **Frequency sweep**: Logarithmically spaced frequency analysis
- **Interactive plots**: SPL and phase response, plus field distribution
- **Theoretical mode calculation**: Compares with analytical room modes
- **Automatic mesh validation**: Warns if resolution is insufficient for frequency range
- **Parallel computation**: Uses all CPU cores via Rayon

## Quick Start

### 2D Mode (Default)

```bash
cargo run --release --bin room_acoustics
```

This simulates a 4m × 2m room in 2D mode.

### 3D Mode

```bash
cargo run --release --bin room_acoustics -- --mode 3d
```

This simulates a 4m × 2m × 3m room in 3D mode.

## Usage

### Command-Line Options

#### Mode Selection
- `--mode <MODE>`: Simulation mode, either `2d` or `3d` (default: `2d`)

#### Frequency Parameters
- `--min-freq <HZ>`: Minimum frequency (default: 50 Hz)
- `--max-freq <HZ>`: Maximum frequency (default: 2000 Hz)
- `--nb-freq <N>`: Number of frequency points, log-spaced (default: 6)

#### Room Dimensions
- `--room-width <M>`: Room width in meters, X dimension (default: 4.0)
- `--room-depth <M>`: Room depth in meters, Y dimension (default: 2.0)
- `--room-height <M>`: Room height in meters, Z dimension, **3D only** (default: 3.0)

#### Source Configuration
- `--num-sources <N>`: Number of sources, 1 or 2 (default: 1)
- `--source1-x <M>`: Source 1 X position in meters (default: 0.3)
- `--source1-y <M>`: Source 1 Y position in meters (default: 0.3)
- `--source1-z <M>`: Source 1 Z position in meters, **3D only** (default: 1.0)
- `--source2-x <M>`: Source 2 X position (if num_sources=2) (default: 3.7)
- `--source2-y <M>`: Source 2 Y position (if num_sources=2) (default: 1.7)
- `--source2-z <M>`: Source 2 Z position, **3D only** (if num_sources=2) (default: 1.0)

#### Measurement Point
- `--measure-x <M>`: Measurement point X in meters (default: 2.0)
- `--measure-y <M>`: Measurement point Y in meters (default: 1.0)
- `--measure-z <M>`: Measurement point Z in meters, **3D only** (default: 1.5)

#### Physical Parameters
- `--absorption <COEF>`: Wall absorption coefficient, 0.0-1.0 (default: 0.1)
- `--grid-resolution <N>`: Grid points per meter (default: 10)

## Examples

### 2D Examples

#### Small Room, Low Frequency
```bash
cargo run --release --bin room_acoustics -- \
  --mode 2d \
  --room-width 3.0 \
  --room-depth 3.0 \
  --max-freq 500 \
  --nb-freq 10
```

#### Stereo Source Setup (2D)
```bash
cargo run --release --bin room_acoustics -- \
  --mode 2d \
  --num-sources 2 \
  --source1-x 1.0 \
  --source1-y 0.5 \
  --source2-x 3.0 \
  --source2-y 0.5 \
  --measure-x 2.0 \
  --measure-y 1.5
```

#### High-Frequency 2D Analysis
```bash
cargo run --release --bin room_acoustics -- \
  --mode 2d \
  --max-freq 2000 \
  --grid-resolution 50 \
  --nb-freq 20
```

### 3D Examples

#### Basic 3D Room
```bash
cargo run --release --bin room_acoustics -- \
  --mode 3d \
  --room-width 4.0 \
  --room-depth 3.0 \
  --room-height 2.5 \
  --max-freq 200 \
  --grid-resolution 5
```

#### Cubic Room (3D)
```bash
cargo run --release --bin room_acoustics -- \
  --mode 3d \
  --room-width 3.0 \
  --room-depth 3.0 \
  --room-height 3.0 \
  --source1-x 1.5 \
  --source1-y 1.5 \
  --source1-z 1.5 \
  --measure-x 1.5 \
  --measure-y 1.5 \
  --measure-z 2.0 \
  --max-freq 300 \
  --grid-resolution 8
```

#### Concert Hall (3D, Low Resolution for Speed)
```bash
cargo run --release --bin room_acoustics -- \
  --mode 3d \
  --room-width 20.0 \
  --room-depth 15.0 \
  --room-height 10.0 \
  --source1-x 2.0 \
  --source1-y 7.5 \
  --source1-z 2.0 \
  --measure-x 10.0 \
  --measure-y 7.5 \
  --measure-z 5.0 \
  --max-freq 100 \
  --grid-resolution 3 \
  --nb-freq 8
```

## Dimension Naming Convention

- **X dimension**: Width (left to right)
- **Y dimension**: Depth (front to back)
- **Z dimension**: Height (bottom to top, 3D only)

### 2D Mode
- Uses X (width) and Y (depth)
- Z dimension is minimal (1 cell)
- Sources and measurement points only use X, Y coordinates

### 3D Mode
- Uses X (width), Y (depth), and Z (height)
- Full 3D wave propagation
- Sources and measurement points use X, Y, Z coordinates

## Mesh Resolution Requirements

⚠️ **IMPORTANT**: The mesh resolution must be adequate for your maximum frequency!

### Rule of Thumb
- **Minimum**: 4 points per wavelength
- **Recommended**: 8+ points per wavelength

### Quick Reference

| Max Freq | Wavelength | Min Resolution | Recommended |
|----------|-----------|----------------|-------------|
| 100 Hz   | 3.43 m    | 2 pts/m        | 3 pts/m     |
| 500 Hz   | 0.686 m   | 6 pts/m        | 12 pts/m    |
| 1000 Hz  | 0.343 m   | 12 pts/m       | 23 pts/m    |
| 2000 Hz  | 0.172 m   | 23 pts/m       | 47 pts/m    |

The simulation will automatically check and warn you if your resolution is insufficient.

## Performance Considerations

### 2D vs 3D
- **2D**: Much faster, suitable for quick analysis or floor-plan-like scenarios
- **3D**: More realistic, but significantly slower (scales with room volume)

### Time Estimates

**2D Mode** (4m × 2m room):
- 10 pts/m, 6 frequencies: ~3-5 seconds
- 50 pts/m, 20 frequencies: ~2-3 minutes

**3D Mode** (4m × 2m × 3m room):
- 5 pts/m, 6 frequencies: ~30-60 seconds
- 10 pts/m, 6 frequencies: ~5-10 minutes

### Tips for Faster Computation
1. Use 2D mode when possible
2. Reduce `--grid-resolution` (but check frequency limits!)
3. Reduce `--nb-freq`
4. Use lower `--max-freq`
5. Make room smaller

## Output Files

Generated in the `plots/` directory:

1. **`room_response_combined.html`** - Combined SPL and phase plot
2. **`room_spl_only.html`** - Sound pressure level only
3. **`room_phase_only.html`** - Phase response only
4. **`room_field_XXXhz.html`** - 2D field distribution
   - 2D mode: Shows full X-Y plane
   - 3D mode: Shows X-Y slice at middle Z height

## Theoretical Room Modes

The simulation calculates and displays theoretical room modes using:

**2D Mode:**
```
f(m,n) = (c/2) × √((m/width)² + (n/depth)²)
```

**3D Mode:**
```
f(m,n,p) = (c/2) × √((m/width)² + (n/depth)² + (p/height)²)
```

where c = 343 m/s (speed of sound in air)

## Parallel Computation

The simulation uses all available CPU cores by default. Control thread count:

```bash
RAYON_NUM_THREADS=4 cargo run --release --bin room_acoustics -- --mode 3d
```

## Troubleshooting

### "WARNING: Only X.X points per wavelength"
**Solution**: Increase `--grid-resolution` to the recommended value shown.

### Slow Performance
- Switch to 2D mode if possible
- Reduce `--grid-resolution` (check frequency limits first!)
- Reduce room size
- Lower `--max-freq`

### "Solver did not converge"
- Reduce `--absorption`
- Lower frequency range
- Check that source and measurement points are inside the room

## Building

From the repository root:

```bash
cargo build --release --bin room_acoustics
```

The binary will be at `target/release/room_acoustics`.

## References

- Speed of sound: 343 m/s (at 20°C)
- Reference pressure: 20 µPa
- Recommended: 4-8 points per wavelength for Helmholtz equation
