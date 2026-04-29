import mouette as M
import sys
import polyscope as ps
import numpy as np

mesh = M.mesh.load(sys.argv[1])

ff = M.framefield.SurfaceFrameField(mesh, "faces", verbose=True, features=False, n_smooth=1)
# ff = M.framefield.PrincipalDirections(mesh, "faces", verbose=True, features=False)
ff_param = M.parametrization.FrameFieldIntegration(ff, scaling=10., verbose=True)
ff_param.run()

# ps.init()
# ps.set_ground_plane_mode("none")

# ps_surf = ps.register_surface_mesh("surf", np.asarray(mesh.vertices), np.asarray(mesh.faces))
# ps.register_curve_network("seams", np.asarray(ff_param.cut_graph.vertices), np.asarray(ff_param.cut_graph.edges), material="flat", color=[1,0,0], radius=0.001)
# ps_surf.add_parameterization_quantity("uv", ff_param.uvs.as_array(len(mesh.face_corners)), defined_on="corners", checker_size=0.5, enabled=True)

# ps.show()

M.mesh.save(ff.export_as_mesh(), "ff.mesh")
M.mesh.save(mesh, "output.geogram_ascii")
M.mesh.save(mesh, "output.obj")
M.mesh.save(ff_param.cut_graph, "cut_graph.mesh")
M.mesh.save(ff_param.flat_mesh, "output_flat.obj")
# M.mesh.save(ff_param._tree.build_tree_as_polyline(), "tree.mesh")