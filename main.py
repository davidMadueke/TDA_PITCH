import os
from TDA_TIME import *
import assets

INPUT_PATH = 'assets/A2_220Hz.wav'

# Get the absolute path to the audio file
INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_PATH)


if __name__ == '__main__':
    signal = TDA(INPUT_PATH, duration=2.0, debug=True)
    pointCloud = signal.pointCloud(windowSize=34,umap_dim=3)
    signal.viewPointCloud3d(pointCloud)

    diag = signal.persistentHomology(showSimplex=True, showBettiNum=True)
    signal.plotHomology(diag, biggerDiagram=False, diagramType=0)
