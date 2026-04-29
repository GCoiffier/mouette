import argparse
import mouette as M
import math

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    args = parser.parse_args()

    mesh = M.mesh.load(args.model)

    scale = mesh.vertices.create_attribute("scale_factor", float)
    scale[mesh.boundary_vertices[0]] = 2.
    scale[mesh.boundary_vertices[4]] = 2.

    bff = M.parametrization.BoundaryFirstFlattening(mesh, bnd_scale_fctr=scale, verbose=True)
    mesh = bff.run()

    M.mesh.save(mesh, "bff.obj")
    M.mesh.save(bff.flat_mesh, "bff_flat.obj")