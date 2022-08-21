import mouette as M

### Point Clouds ###

def pc_sample1():
    m = M.mesh.new_point_cloud()
    m.vertices += [
        M.Vec(0., 0., 0.),
        M.Vec(1., 0., 0.),
        M.Vec(0., 1., 0.),
        M.Vec(0., 0., 1.),
    ]

def pc_random10():
    m = M.mesh.new_point_cloud()
    m.vertices += [
        M.Vec.random(3) for _ in range(10)
    ]

def pc_grid():
    m = M.mesh.new_point_cloud()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                m.vertices.append(M.Vec(float(x), float(y), float(z)))
    return m

point_clouds = [
    pc_sample1(),
    pc_random10(),
    pc_grid()
]