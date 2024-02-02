# TDA_PITCH
A Pitch Detection Algorithm combining ML and Topological Data Analysis (TDA) approaches. Developed as part of my MEng EE Thesis

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install all of the required dependencies this project use pip.
```bash
pip install -r requirements.txt

```
## Usage
To use the TDA class simply navigate to main.py and add the audio file in question to the assets folder.

```python
import os
from TDA import *
import assets

INPUT_PATH = 'assets/trial1.wav'

# Get the absolute path to the audio file
INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_PATH)


if __name__ == '__main__':
    signal = TDA(INPUT_PATH, duration=10.0, debug=True)
    pointCloud = signal.pointCloud(windowSize=1,umap_dim=3)
    signal.viewPointCloud3d(pointCloud)

    diag = signal.persistentHomology(showSimplex=True, showBettiNum=True)
    signal.plotHomology(diag, biggerDiagram=False, diagramType=1)

```