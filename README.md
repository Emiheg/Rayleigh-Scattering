# Rayleigh Scattering Studio

An interactive Python simulation of Rayleigh scattering and how it changes the color of the sky as the sun moves through the atmosphere.

The app uses Matplotlib sliders to adjust the sun position, atmosphere parameters, and the sampled canvas line. It shows the current and future sky gradients side by side, plus a larger generated animation preview.

<img width="288" height="162" alt="Untitled" src="https://github.com/user-attachments/assets/d4cd7f55-112b-4b24-9d85-a19186625d06" />

<img width="288" height="162" alt="Adobe Express - sunrise" src="https://github.com/user-attachments/assets/204ac108-fd1e-446e-bc58-006799355b15" />

<img width="288" height="162" alt="sunset_01 (1)" src="https://github.com/user-attachments/assets/d98a49f9-8d90-43e4-9628-4d6faa9afcaf" />




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
