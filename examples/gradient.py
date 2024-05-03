import mouette as M
from mouette import geometry as geom

import scipy.sparse as sp
import cmath
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", type=str, help="path to the input mesh")
    parser.add_argument("-outp", "--outp", default="output.geogram_ascii")
    parser.add_argument("-l", "--eigen", type=float, default=100., help="eigenvalue")
    parser.add_argument("-signal", type=str, choices=["eigen", "z"])

    args = parser.parse_args()

    OUTPUT_FILE = args.outp
    PREV = OUTPUT_FILE.split(".")
    OUTPUT_GRAD = PREV[0] + "_grad." + PREV[1]
    
    print("Load mesh")
    mesh = M.mesh.load(args.input_mesh)
    connection = M.processing.SurfaceConnectionFaces(mesh)

    if args.signal == "eigen":
        print("Compute Laplacian")
        lap = M.operators.laplacian(mesh)
        A = M.operators.area_weight_matrix(mesh)

        print("Compute eigenvector of laplacian near eigenvalue ", args.eigen)
        eigvalue, signal = sp.linalg.eigsh(lap, M=A, k=1, sigma=args.eigen)
        print("Found eigenvalue:", eigvalue[0])

    elif args.signal =="z":
        signal = np.zeros(len(mesh.vertices))
        for iV,V in enumerate(mesh.vertices):
            signal[iV] = V.z
        signal = signal[:,np.newaxis]

    print("Compute gradient")
    grad = M.operators.gradient(mesh, connection)
    signal_G = grad @ signal

    print("Output results")
    signal_attr = mesh.vertices.create_attribute("f", float, dense=True)
    signal_attr._data = signal

    grad_mesh = M.mesh.RawMeshData()
    L = M.attributes.mean_edge_length(mesh)/3
    for id_face, face in enumerate(mesh.faces):
        basis,Y = connection.base(id_face)
        normal = geom.cross(basis,Y)
        pA,pB,pC = (mesh.vertices[_v] for _v in face)
        angle = cmath.phase(signal_G[id_face])
        bary = (pA+pB+pC)/3 # reference point for display
        direction = geom.rotate_around_axis(basis, normal, angle)
        pt = bary + abs(signal_G[id_face])*L*direction
        grad_mesh.vertices += [bary, pt]
        grad_mesh.edges.append((2*id_face, 2*id_face+1))

    M.mesh.save(mesh, OUTPUT_FILE)
    M.mesh.save(grad_mesh, OUTPUT_GRAD)

