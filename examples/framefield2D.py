import argparse
import mouette as M
from mouette.processing import SurfaceFrameField, PrincipalDirections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", type=str, help="path to the input mesh")
    parser.add_argument("-outp", "--outp", default="output/output.geogram_ascii")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-elem", "--elements", type=str, choices=["vertices", "faces"])
    parser.add_argument("-order", "--order", type=int, default=4)
    parser.add_argument("-feat", "--features", action="store_true")
    parser.add_argument("-curv", "--curvature", action="store_true")
    parser.add_argument("-cadff", "--cadff", action="store_true")
    parser.add_argument("-singus", "--singular-indices", nargs="+", help="list of alternating elem_id singu_id")
    parser.add_argument("-n", "--n-smooth", default=0, type=int)
    parser.add_argument("-alpha", "--alpha", default=None, type=float)
    args = parser.parse_args()

    OUTPUT_FILE = args.outp
    PREV = OUTPUT_FILE.split(".")
    OUTPUT_FF  = PREV[0] + "_FF." + PREV[1]

    mesh = M.mesh.load(args.input_mesh)
    
    ## Handle provided singularities
    singus = None
    if args.singular_indices is not None:
        assert len(args.singular_indices)%2 == 0
        singus = mesh.vertices.create_attribute("singuls", float) if args.elem == "faces" else mesh.faces.create_attribute("singuls", float)
        for i in range(len(args.singular_indices)//2):
            singus[args.singular_indices[2*i]] = args.singular_indices[2*i+1]

    if args.curvature:
        ff = PrincipalDirections(mesh, args.elements, args.features, args.verbose, n_smooth=args.n_smooth, smooth_attach_weight=args.alpha)
    else:
        ff = SurfaceFrameField(mesh, args.elements, args.order, args.features, verbose=args.verbose, n_smooth=args.n_smooth, smooth_attach_weight=args.alpha, cad_correction=args.cadff, singularity_indices=singus)

    ff.run()
    ff.flag_singularities()
    M.mesh.save(mesh, OUTPUT_FILE)
    M.mesh.save(ff.export_as_mesh(), OUTPUT_FF)
