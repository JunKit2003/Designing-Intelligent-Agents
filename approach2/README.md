# Adaptive Traffic Signal Control using Reinforcement Learning in SUMO (Approach 2)

## Overview

This project implements and evaluates an Adaptive Traffic Signal Control (ATSC) system using Deep Reinforcement Learning (RL) within the SUMO (Simulation of Urban MObility) traffic simulator. The primary goal is to optimise traffic flow and reduce congestion in urban intersections compared to traditional fixed-time controllers.

This specific implementation (referred to as "Approach 2" in development) focuses on two scenarios based on real-world layouts: **Acosta** and **Pasubio**. It utilises a layout-specific approach where intersections are grouped based on proximity and traffic interdependence. A single, shared **Noisy Double Deep Q-Network (DQN)** agent is trained to control the traffic signals for each group of intersections.

The effectiveness of the RL agent is evaluated against a baseline fixed-time traffic signal controller by comparing metrics such as average vehicle waiting time, queue length, and travel time. Results demonstrate significant improvements in traffic efficiency for both the Acosta and Pasubio scenarios.

---

## Software Requirements

- **SUMO:** A recent version (e.g., 1.10.0 or later). Ensure SUMO binaries are correctly installed and accessible.
- **Python:** Version 3.8 or later recommended.
- **TensorFlow:** For the Deep Learning model (preferably version 2.x).
- **Traci (Traffic Control Interface):** Python library for interacting with SUMO (usually included with SUMO installation).
- **NumPy:** For numerical operations.
- **Pandas:** For data handling (metrics).
- **Matplotlib:** For plotting results (optional, used in utility scripts).
- **Other standard Python libraries:** `os`, `sys`, `argparse`, `datetime`, `random`, `json`, `csv`.

---

## Setup Instructions

### 1. Organise Project Files

Ensure all provided project files (`src/`, `acosta/`, `pasubio/`, etc.) are organised within a main project directory on your local machine.

---

### 2. SUMO Installation

- Download and install SUMO from the [official website](https://www.eclipse.org/sumo/).
- Add the SUMO installation's `bin` directory to your system's `PATH`, **OR** set the `SUMO_HOME` environment variable pointing to your SUMO installation directory (e.g., `/path/to/sumo-1.10.0`).

**Example (Linux/macOS):**

```bash
export SUMO_HOME="/path/to/sumo-1.10.0"
export PATH="$PATH:$SUMO_HOME/bin"
```

**Example (Windows - Command Prompt):**

```cmd
set SUMO_HOME="C:\path\to\sumo-1.10.0"
set PATH=%PATH%;%SUMO_HOME%\bin
```

> **Note:** You might want to set these environment variables permanently depending on your OS settings.

---

### 3. Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

---

### 4. Install Dependencies

After activating your virtual environment, install the required Python packages:

```bash
pip install -r requirements.txt
```

---

### 5. Generating `requirements.txt` (Optional)

If the `requirements.txt` file is missing, you can generate it manually. Below is an example of its content:

```txt
tensorflow>=2.8.0
numpy>=1.20.0
pandas>=1.2.0
matplotlib>=3.3.0
traci>=1.10.0
scipy>=1.6.0
```

You can create this file by copying the above text into a file named `requirements.txt` in your project root directory.

Alternatively, you can generate it automatically after setting up your environment and installing necessary packages using tools like [pipreqs](https://pypi.org/project/pipreqs/).

---

## Project Structure

```
/project_root
    /src
        (Python source files, agents, utilities)
    /acosta
        (Scenario-specific SUMO files)
    /pasubio
        (Scenario-specific SUMO files)
    requirements.txt
    README.md
```

---

## How to Run

After completing the setup:

1. Activate the virtual environment (if not already active).
2. Ensure that the SUMO environment variables are correctly set.
3. Navigate to the `src/` directory.
4. Run the training or evaluation script. Example:

```bash
python train.py --scenario acosta
```

or

```bash
python evaluate.py --scenario pasubio
```

Command-line arguments will vary depending on your script's setup (e.g., scenario name, training steps, evaluation mode, model saving options).

---

## Notes

- Training Deep RL models can take significant time depending on the scenario complexity and hardware.
- To speed up development/testing, you can use reduced traffic flow files or smaller simulation time steps during debugging.
- Use `matplotlib` or other plotting libraries to visualise the performance metrics (waiting times, queue lengths, etc.) if needed.

---

## Acknowledgements

- SUMO Traffic Simulator ([Eclipse SUMO](https://www.eclipse.org/sumo/))
- TensorFlow
- Open-source Deep RL algorithm implementations that inspired parts of the Noisy Double DQN setup.

---
