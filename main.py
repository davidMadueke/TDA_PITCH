import os
from TDA_TIME import *
import assets


INPUT_FILENAME = 'assets/trial3.wav'

# Get the absolute path to the audio file
INPUT_PATH = os.path.join(os.path.dirname(__file__), INPUT_FILENAME)


if __name__ == '__main__':
    signal = TDA(INPUT_PATH, duration=2.0, debug=True, shortFileName=INPUT_FILENAME)
    pointCloud = signal.pointCloud(windowSize=34,umap_dim=3)
    signal.viewPointCloud3d(pointCloud)

    diag = signal.persistentHomology(showSimplex=True, showBettiNum=True, filter=None)
    signal.plotHomology(diag, biggerDiagram=False, diagramType=0)
