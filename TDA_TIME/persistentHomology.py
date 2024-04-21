import gudhi as gd


def homology_from_3dcloud(cloud):
    alpha_asc = create_alpha_complex(cloud)
    diag = alpha_asc.persistence()
    return diag


def create_alpha_complex(point_cloud):
    alpha_complex = gd.AlphaComplex(points=point_cloud)
    alpha_asc = alpha_complex.create_simplex_tree()
    return alpha_asc



