import gudhi as gd
import numpy as np
import time
import scipy
from matplotlib import pyplot as plt
import plotly.io as pio
from TDA_PITCH.TDA.preprocessing import *
from TDA_PITCH.TDA.dataVisualisation import *
from TDA_PITCH.TDA.pointCloud import *
from TDA_PITCH.TDA.persistentHomology import *


class TDA:
    # Initialising this object will preprocess the audio file in the format needed for the TDA
    def __init__(self, filename,
                 startTime: float = 0, duration: float = 30,
                 downsampleFactor: int = 5, debug: bool =True):
        self.startTime = startTime
        self.duration = duration
        self.downsampleFactor = downsampleFactor
        self.debug = debug

        if self.debug:
            print("Generating Audio Signal")
        signal, self.sampleRate = import_audio_signal(filename,
                                                      startTime=self.startTime,
                                                      duration=self.duration)

        signal_downsampled = signal_downSample(signal, self.downsampleFactor)
        signal_bandpass = butter_bandpass_filter(signal_downsampled, 500, 1500, self.sampleRate)

        self.signal = signal_bandpass

        self.cloud = None
        self.alpha_asc = None
        self.betti_nums = None
        self.diag = None

    # This method will take an incoming signal and generate a resulting point cloud using Taken's Embedding Thm
    def pointCloud(self, windowSize, umap_dim=2):
        if self.debug:
            print("Generating Point Cloud")
        cloud = cloud_from_signal(self.signal, windowSize, umap_dim, embed_dim=3, debug=self.debug)
        self.cloud = normalize_3dcloud(cloud)
        return self.cloud

    # This method allows for the point cloud to be visualised
    @staticmethod
    def viewPointCloud3d(pointCloudVector):
        figure = pointCloud3d(pointCloudVector)
        pio.show(figure)

    # Calling this method will perform persistent homology on this point cloud returning a persistence diagram
    def persistentHomology(self,
                           showSimplex: bool = False, showBettiNum: bool = False,
                           filter=1):

        if self.debug:
            print("Creating the Alpha Complex")
        self.alpha_asc = create_alpha_complex(self.cloud)

        if showSimplex:
            getSimplex(self.alpha_asc, fullSTree=False)

        if self.debug:
            print("Generating Persistence Diagram")

        self.diag = self.alpha_asc.persistence()

        if showBettiNum:
            self.betti_nums = get_betti_num(self.alpha_asc)
        if filter == 1:
            self.diag = [(a, (b, c)) for (a, (b, c)) in self.diag if b > 0.00027 and c - b > 0.002]
        return self.diag

    # Calling this method will plot the persistence diagram for this audio file and its accompanying betti numbers and
    # the number of simplices.
    @staticmethod
    def plotHomology(persistenceDiagram, biggerDiagram: bool = False, diagramType: int = 0):
        if biggerDiagram:
            print_bigger_diagram(persistenceDiagram)
        else:
            print_diagram(persistenceDiagram, type=diagramType)  # type=0 diag , type = 1 barcode

