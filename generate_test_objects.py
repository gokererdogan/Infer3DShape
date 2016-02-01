"""
Inferring 3D Shape from 2D Images

This script generates simple test objects.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
from shape import *
from voxel_based_shape import VoxelBasedShape, Voxel, EMPTY_VOXEL, FULL_VOXEL


def get_viewpoint(angle):
    x = 2.0 * np.cos(angle / 180.0 * np.pi)
    y = 2.0 * np.sin(angle / 180.0 * np.pi)
    viewpoint = [(x, y, 2.0)]
    return viewpoint

if __name__ == "__main__":
    import vision_forward_model as vfm
    fwm = vfm.VisionForwardModel(render_size=(200, 200))
    # test 1
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.5, .75/2, .75/2])),
             CuboidPrimitive(np.array([.75/2, 0.0, 0.0]), np.array([.25, .25, .25]))]

    viewpoint = get_viewpoint(262)
    h = Shape(fwm, parts=parts, viewpoint=viewpoint)
    # fwm._view(h)
    img = fwm.render(h)
    np.save('./data/test1_single_view.npy', img)
    fwm.save_render('./data/test1_single_view.png', h)

    # test 2
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.5, .75/2, .75/2])),
             CuboidPrimitive(np.array([.45, 0.0, 0.0]), np.array([.4, .25, .25])),
             CuboidPrimitive(np.array([0.0, 0.0, 0.75/2]), np.array([0.25/2, 0.35/2, .75/2])),
             CuboidPrimitive(np.array([0.0, 0.2, 0.75/2]), np.array([.1, .45/2, .25/2]))]

    viewpoint = get_viewpoint(296)
    h = Shape(fwm, parts=parts, viewpoint=viewpoint)
    # fwm._view(h)
    img = fwm.render(h)
    np.save('./data/test2_single_view.npy', img)
    fwm.save_render('./data/test2_single_view.png', h)

    # test 3
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.2, 0.45])),
             CuboidPrimitive(np.array([.35/2, 0.0, 0.10]), np.array([0.15, 0.15, 0.15])),
             CuboidPrimitive(np.array([0.3, 0.0, -0.45/2]), np.array([0.4, 0.2, 0.2])),
             CuboidPrimitive(np.array([0.85/2, 0.0, 0.05/2]), np.array([0.15, 0.15, 0.3])),
             CuboidPrimitive(np.array([-0.25, 0.0, 0.15]), np.array([0.3, 0.3, 0.3])),
             CuboidPrimitive(np.array([-0.25, -0.25, 0.2]), np.array([0.1, 0.2, 0.1]))]

    viewpoint = get_viewpoint(7)
    h = Shape(fwm, parts=parts, viewpoint=viewpoint)
    # fwm._view(h)
    img = fwm.render(h)
    np.save('./data/test3_single_view.npy', img)
    fwm.save_render('./data/test3_single_view.png', h)


    # test 4
    viewpoint = get_viewpoint(49)
    v = VoxelBasedShape(fwm, viewpoint=viewpoint)
    for vx in v.voxel.subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL

    v.voxel.subvoxels[0, 0, 0] = Voxel.get_random_voxel(v.voxel.subvoxels[0, 0, 0].origin,
                                                        v.voxel.subvoxels[0, 0, 0].depth)
    for vx in v.voxel.subvoxels[0, 0, 0].subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL
    v.voxel.subvoxels[0, 0, 0].subvoxels[0, 1, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 0, 0].subvoxels[1, 1, 0].status = FULL_VOXEL

    v.voxel.subvoxels[0, 1, 0] = Voxel.get_random_voxel(v.voxel.subvoxels[0, 1, 0].origin,
                                                        v.voxel.subvoxels[0, 1, 0].depth)
    for vx in v.voxel.subvoxels[0, 1, 0].subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[0, 0, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[0, 1, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 0, 1].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].status = FULL_VOXEL
    img = fwm.render(v)
    np.save('./data/test4_single_view.npy', img)
    fwm.save_render('./data/test4_single_view.png', v)

    # test 5
    viewpoint = get_viewpoint(349)
    v = VoxelBasedShape(fwm, viewpoint=viewpoint)
    for vx in v.voxel.subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL

    v.voxel.subvoxels[0, 0, 0] = Voxel.get_random_voxel(v.voxel.subvoxels[0, 0, 0].origin,
                                                        v.voxel.subvoxels[0, 0, 0].depth)
    for vx in v.voxel.subvoxels[0, 0, 0].subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL
    v.voxel.subvoxels[0, 0, 0].subvoxels[0, 1, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 0, 0].subvoxels[1, 1, 0].status = FULL_VOXEL

    v.voxel.subvoxels[0, 1, 0] = Voxel.get_random_voxel(v.voxel.subvoxels[0, 1, 0].origin,
                                                        v.voxel.subvoxels[0, 1, 0].depth)
    for vx in v.voxel.subvoxels[0, 1, 0].subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[0, 0, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[0, 1, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 0, 1].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0] = Voxel.get_random_voxel(v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].origin,
                                                                           v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].depth)
    for vx in v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].subvoxels:
        for vy in vx:
            for vz in vy:
                vz.status = EMPTY_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].subvoxels[0, 1, 1].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].subvoxels[1, 1, 0].status = FULL_VOXEL
    v.voxel.subvoxels[0, 1, 0].subvoxels[1, 1, 0].subvoxels[1, 0, 1].status = FULL_VOXEL

    img = fwm.render(v)
    np.save('./data/test5_single_view.npy', img)
    fwm.save_render('./data/test5_single_view.png', v)

