import gudhi as gd


def homology_from_3dcloud(cloud, max_alpha):
    alpha_asc = create_alpha_complex(cloud, max_alpha_sqrt=max_alpha)
    diag = alpha_asc.persistence()
    return diag


def create_alpha_complex(point_cloud):
    alpha_complex = gd.AlphaComplex(points=point_cloud)
    alpha_asc = alpha_complex.create_simplex_tree()
    return alpha_asc



