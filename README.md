# Rayleigh Scattering Studio

An interactive Python simulation of Rayleigh scattering and how it changes the color of the sky as the sun moves through the atmosphere.

The app uses Matplotlib sliders to adjust the sun position, atmosphere parameters, and the sampled canvas line. It shows the current and future sky gradients side by side, plus a larger generated animation preview.

<img width="1040" height="790" alt="Rayleigh scattering simulator" src="https://github.com/user-attachments/assets/91465f30-f2e5-414d-b87a-ea0053f871db" />

## Features

- Move the sun around a simplified atmosphere.
- Tune scattering intensity and density falloff.
- Adjust the canvas sampling line and bend.
- Preview the sky color before and after a chosen rotation.
- Generate an in-window animation of the transition.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Run

```bash
python main.py
```

Use the sliders to change the simulation. Click **Generate animation** to render the animated preview in the right panel.

## Project Structure

- `main.py` - builds the UI and connects the simulation controls.
- `canvas.py` - scattering calculations and canvas rendering helpers.
- `SunClass.py` - sun position, color, and sampling-line state.
- `video.py` - generated animation preview.
- `style.py` - layout and styling helpers.
- `config.py` - simulation constants.
