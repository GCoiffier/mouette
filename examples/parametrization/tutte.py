import argparse
import mouette as M

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("-bnd", "--boundary-mode", type=str, choices=["square", "circle"], default="circle")
    parser.add_argument("-cotan", "--cotangent", action="store_true", help="whether to use cotangent weights")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    mesh = M.mesh.load(args.model)

    tutte = M.parametrization.TutteEmbedding(mesh, args.boundary_mode, use_cotan=args.cotangent, verbose=args.verbose)
    tutte.run()

    M.mesh.save(mesh, "tutte_model.obj")
    M.mesh.save(tutte.flat_mesh, "tutte_flat.obj")