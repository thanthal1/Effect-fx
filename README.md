## Overview

This project pushes Python-based computer vision and rendering toward real-time performance limits. It combines optimized numerical computation, conditional GPU acceleration, and asynchronous input processing.

Effects are generated independently of the camera feed, allowing external compositing (e.g., in OBS). This separation improves both flexibility and performance.

![Lightning Coming out of hand](https://github.com/thanthal1/Effect-fx/blob/master/Example.png)


---

## Features

### Effects
- Fire (particle system + heat simulation)
- Frost (convolution-based propagation)
- Lightning (procedural branching)
- Smoke (particle-based, upward flow)
- Plasma (animated mathematical field)

### Tracking
- Hand tracking with gesture-aware anchoring
- Face tracking using facial landmarks

### Performance
- Vectorized particle updates (NumPy)
- Convolution kernels via OpenCV (no repeated array shifts)
- LUT-based color grading
- Conditional CUDA acceleration with CPU/OpenCL fallback
- Region-of-interest (ROI) rendering
- Skips inactive effects
- Reduced-resolution simulation buffers

### System Design
- Asynchronous tracking (non-blocking render loop)
- Modular effect pipeline
- Web-based control panel

---

## Architecture

The system is divided into three components:

### Input
Hand and face tracking run asynchronously in a separate thread to avoid blocking rendering.

### Simulation
Effects are computed using NumPy and OpenCV. Buffers operate at reduced resolution to minimize cost.

### Rendering
Effects are composited into the output frame using optimized blending and optional GPU acceleration.

---

## Performance

Designed for multi-core CPUs and optional GPU acceleration.

Key strategies:
- Early downscaling of camera input
- Vectorized math instead of Python loops
- Conditional GPU usage to avoid transfer overhead
- ROI-based computation
- Skipping unnecessary work

Performance varies by hardware and effect complexity.

---

## OBS Integration

Intended to be used as an overlay source.

Typical setup:
1. Run the program
2. Add output as a window capture


---

## Controls

A local Flask-based control panel allows:

- Effect selection
- Input source switching (hand / face)
- Intensity adjustment
- Particle scaling
- FPS and acceleration monitoring

---

## Requirements

- Python 3.11(havent tested others)
- OpenCV (CUDA optional)
- NumPy
- cvzone (MediaPipe wrapper)
- Flask

GPU acceleration requires an OpenCV build with CUDA support and a compatible NVIDIA GPU.

---


## License
Do whatever you want with this but heres that in more formal language 

Zero-Clause BSD

Permission to use, copy, modify, and/or distribute this software for
any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

