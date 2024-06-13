import argparse
import mouette as M
from mouette import framefield

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", type=str, help="path to the input mesh")
    parser.add_argument("-outp", "--outp", default="output.geogram_ascii")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-elem", "--elements", type=str, choices=["vertices", "cells"], required=True)
    parser.add_argument("-feat", "--features", action="store_true")
    parser.add_argument("-n", "--n-smooth", default=0, type=int)
    parser.add_argument("-alpha", "--alpha", default=None, type=float)
    args = parser.parse_args()

    OUTPUT_FILE = args.outp
    PREV = OUTPUT_FILE.split(".")
    OUTPUT_FF  = PREV[0] + "_FF." + PREV[1]

    mesh = M.mesh.load(args.input_mesh)
    
    ## Handle provided singularities
    
    ff = framefield.VolumeFrameField(mesh, args.elements, args.features, 
        n_smooth=args.n_smooth, smooth_attach_weight=args.alpha, verbose=args.verbose)
    ff.run()
    ff.flag_singularities()
    M.mesh.save(mesh, OUTPUT_FILE)
    M.mesh.save(ff.export_as_mesh(), OUTPUT_FF)
