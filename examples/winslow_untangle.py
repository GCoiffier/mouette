import argparse
import mouette as M
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("-l", "--lmbd", type=float, default=1.0, help="weighting term for g")
    parser.add_argument("-n", "--n-iter", type=int, default=10_000, help="number of inner iterations")
    parser.add_argument("-m", "--n-eps-update", type=int, default=10)
    args = parser.parse_args()

    mesh = M.mesh.load(args.model)
    if not mesh.is_triangular():
        raise Exception("Mesh is not triangular")
    
    # Create a flat mesh 
    uv_init = mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
    for v in mesh.id_vertices:
        uv_init[v] = [mesh.vertices[v][0], mesh.vertices[v][2]] # flatten y coordinate
        # uv_init[v] = [mesh.vertices[v][0], mesh.vertices[v][2] + mesh.vertices[v][1]] # even more broken initialization

    uv_init_corners = mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
    M.attributes.scatter_vertices_to_corners(mesh, uv_init, uv_init_corners)

    dist_init = M.parametrization.ParamDistortion(mesh)()
    print("Initial Mapping Distortion:", dist_init.summary)
    M.mesh.save(mesh, "init_mesh.obj")
    M.mesh.save(mesh, "init_mesh.geogram_ascii")

    # Run the untangler
    untangler = M.parametrization.WinslowInjectiveEmbedding(mesh, uv_init, lmbd=args.lmbd)
    M.mesh.save(untangler.flat_mesh, "init_flat.obj")
    
    untangler.run()
    uv_init_corners = M.attributes.scatter_vertices_to_corners(mesh, untangler.uvs, uv_init_corners)
    dist_final = M.parametrization.ParamDistortion(mesh)()
    print("Final Mapping Distortion:", dist_final.summary)
    M.mesh.save(mesh, "final_mesh.obj")
    M.mesh.save(mesh, "final_mesh.geogram_ascii")
    M.mesh.save(untangler.flat_mesh, "final_flat.obj")

    

