from gtda.time_series import SingleTakensEmbedding
import numpy as np
import umap
from matplotlib import pyplot as plt


def cloud_from_signal(signal, window_size, umap_dim=2, embed_dim=5, debug: bool = True):
    # Create the Transformer which will optimise the time delay and embedding dimension
    STE = SingleTakensEmbedding(parameters_type='search',
                                dimension=embed_dim,
                                time_delay=window_size,
                                n_jobs=None)  # Use default values from gtda docs
    # Perform the Takens' Embedding
    signal_embedded = STE.fit_transform(signal)
    cloud = dim_reduction(signal_embedded, umap_dim, debug=debug)
    return cloud


def dim_reduction(X_transform, umap_dim, n_neighbors=10, debug=False):
    red = umap.UMAP(a=None, angular_rp_forest=False, b=None,
                    force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
                    local_connectivity=1.0, low_memory=False, metric='euclidean',
                    metric_kwds=None, min_dist=0.5, n_components=umap_dim, n_epochs=None,
                    n_neighbors=n_neighbors, negative_sample_rate=5, output_metric='euclidean',
                    output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
                    set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
                    target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
                    transform_queue_size=4.0, transform_seed=42, unique=False, verbose=debug)
    return red.fit_transform(X_transform)


def normalize_3dcloud(cloud):
    cloud = (cloud - np.mean(cloud)) / np.max(np.max(abs(cloud)))
    return cloud



