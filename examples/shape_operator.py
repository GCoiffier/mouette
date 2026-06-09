import polyscope as ps
import sys
import numpy as np

import mouette as M
from mouette.processing import ShapeOperator

if __name__ == "__main__":
    input_mesh = M.mesh.load(sys.argv[1])
    input_mesh = M.transform.fit_into_unit_cube(input_mesh)
    shape_op = ShapeOperator(input_mesh)
    shape_op.run()

    ps.init()
    ps_mesh = ps.register_surface_mesh("surface", np.asarray(input_mesh.vertices), np.asarray(input_mesh.faces))
    GC = shape_op.gaussian_curvature.as_array()
    print(np.sum(GC))
    ps_mesh.add_scalar_quantity("gaussian_curvature", GC)
    ps_mesh.add_scalar_quantity("mean_curvature", shape_op.mean_curvature.as_array())
    ps.show()
