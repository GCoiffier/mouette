import mouette as M
import numpy as np
import polyscope as ps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_mesh", type=str, help="path to the input mesh")
parser.add_argument("-s", "--sources", nargs="+", help="list of source vertices", type=int, default=[0])
parser.add_argument("-bnd", action="store_true", help="make all boundary vertices sources")
parser.add_argument("-d", "--diffuse", type=float, default=1., help="diffusion coefficient")
args = parser.parse_args()

mesh = M.mesh.load(args.input_mesh)

ps.init()
ps.set_ground_plane_mode("none")

heat_solver = M.processing.HeatMethodDistance(mesh, save_on_mesh=False, verbose=True, diffuse_coeff=args.diffuse)

if args.bnd:
    distance, gradients = heat_solver.get_distance_to_boundary(return_gradients=True)
else:
    distance, gradients = heat_solver.get_distance(args.sources, return_gradients=True)

ps_mesh = ps.register_surface_mesh("surface", np.asarray(mesh.vertices), np.asarray(mesh.faces))
grads = np.stack((np.real(gradients), np.imag(gradients)), axis=-1)
ps_mesh.add_tangent_vector_quantity("grad", grads, heat_solver.conn.bX, heat_solver.conn.bY, defined_on="faces", enabled=False)
ps_mesh.add_scalar_quantity("dist", distance, enabled=True, cmap="reds", isolines_enabled=True)
if not args.bnd:
    ps.register_point_cloud("sources", np.asarray([mesh.vertices[x] for x in args.sources]), color=[0.,0.,0.])
ps.show()