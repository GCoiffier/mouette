import argparse
import mouette as M
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("-mode", "--mode", type=str, choices=["alternate", "bfgs"], default="bfgs")
    args = parser.parse_args()

    mesh = M.mesh.load(args.model)
    if M.attributes.euler_characteristic(mesh)!=1:
        raise Exception("Mesh is not a disk")
    if not mesh.is_triangular():
        raise Exception("Mesh is not triangular")
    
    # Initialize with Tutte's embedding
    bnd = np.array([[mesh.vertices[p][0], mesh.vertices[p][2]] for p in mesh.boundary_vertices]) # recover the boundary
    tutte = M.parametrization.TutteEmbedding(mesh, use_cotan=True, verbose=True, custom_boundary=bnd, save_on_corners=False)()
    uv_init = tutte.uvs
    uv_init_corners = mesh.face_corners.create_attribute("uv_coords", float, 2, dense=True)
    M.attributes.scatter_vertices_to_corners(mesh, uv_init, uv_init_corners)

    # Compute mapping's distortion and output initial state
    dist_init = M.parametrization.ParamDistortion(mesh)()
    print("Initial Mapping Distortion:", dist_init.summary)
    M.mesh.save(mesh, "init_mesh.obj")
    M.mesh.save(mesh, "init_mesh.geogram_ascii")

    # Run the cotan embedding
    untangler = M.parametrization.CotanEmbedding(mesh, uv_init, verbose=True, solver_verbose=True, mode=args.mode)
    M.mesh.save(untangler.flat_mesh, "init_flat.obj")
    untangler.run()

    # Output final state
    uv_init_corners = M.attributes.scatter_vertices_to_corners(mesh, untangler.uvs, uv_init_corners)
    dist_final = M.parametrization.ParamDistortion(mesh)()
    print("Final Mapping Distortion:", dist_final.summary)
    M.mesh.save(mesh, "final_mesh.obj")
    M.mesh.save(mesh, "final_mesh.geogram_ascii")
    M.mesh.save(untangler.flat_mesh, "final_flat.obj")

    

