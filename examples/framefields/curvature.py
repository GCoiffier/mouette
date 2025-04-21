import argparse
import mouette as M
from mouette import framefield

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", type=str, help="path to the input mesh")
    parser.add_argument("-outp", "--outp", default="output.geogram_ascii")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-elem", "--elements", type=str, choices=["vertices", "faces"])
    parser.add_argument("-ps", "--patch-size", type=int, default=2)
    parser.add_argument("-ct", "--confidence-threshold", type=float, default=0.5)
    parser.add_argument("-st", "--smooth-threshold", type=float, default=0.7)
    parser.add_argument("-n", "--n-smooth", type=int, default=1)

    parser.add_argument("-feat", "--features", action="store_true")
    args = parser.parse_args()

    OUTPUT_FILE = args.outp
    PREV = OUTPUT_FILE.split(".")
    OUTPUT_FF  = PREV[0] + "_FF." + PREV[1]

    mesh = M.mesh.load(args.input_mesh)    
    ff = framefield.PrincipalDirections(mesh, args.elements, features=args.features, verbose=args.verbose, n_smooth=args.n_smooth, smooth_threshold=args.smooth_threshold, confidence_threshold=args.confidence_threshold, patch_size=args.patch_size)
    ff.run()
    ff.flag_singularities()
    M.mesh.save(mesh, OUTPUT_FILE)
    M.mesh.save(ff.export_as_mesh(), OUTPUT_FF)


