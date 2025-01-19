import mouette as M

sphere = M.procedural.icosphere()

ff = M.framefield.SurfaceFrameField(sphere, "faces")
ff_param = M.parametrization.FrameFieldIntegration(ff, verbose=True)
ff_param.run()

M.mesh.save(sphere, "sphere.obj")
M.mesh.save(ff_param.flat_mesh, "sphere_flat.obj")