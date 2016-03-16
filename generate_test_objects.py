"""
Inferring 3D Shape from 2D Images

This script generates simple test objects.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import cPickle as pkl

from shape import *
from voxel_based_shape import VoxelBasedShape, Voxel, EMPTY_VOXEL, FULL_VOXEL
import geometry_3d


def get_viewpoint(angle):
    return np.sqrt(8.0), angle, 45.0

if __name__ == "__main__":
    """
    import vision_forward_model as vfm
    fwm = vfm.VisionForwardModel(render_size=(200, 200))

    # test 1
    parts = [CuboidPrimitive(np.array([0.0, 0.0, 0.0]), np.array([0.5, .75/2, .75/2])),
             CuboidPrimitive(np.array([.75/2, 0.0, 0.0]), np.array([.25, .25, .25]))]

    viewpoint = get_viewpoint(262)
    h = Shape(fwm, parts=parts, viewpoint=viewpoint)

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
    """

    # test 6
    import vision_forward_model as vfm
    fwm = vfm.VisionForwardModel(render_size=(200, 200), custom_lighting=False)

    import paperclip_shape as pc_shape
    joint_positions = np.array([[-0.54028188,  0.41323616,  0.19661439],
                                   [-0.20661808,  0.51499101, -0.1030599 ],
                                   [-0.12298621,  0.        ,  0.        ],
                                   [ 0.12298621,  0.        ,  0.        ],
                                   [ 0.84088651,  0.1529245 , -0.10760027],
                                   [ 0.4336905 , -0.07415333, -1.01534363]])

    pc = pc_shape.PaperClipShape(forward_model=fwm, viewpoint=[get_viewpoint(45.0)], joint_positions=joint_positions,
                                 min_joints=2, max_joints=8, mid_segment_id=2)
    img = fwm.render(pc)
    np.save('./data/test6_single_view.npy', img)
    fwm.save_render('./data/test6_single_view.png', pc)
    pkl.dump(pc, open('./data/test6.pkl', 'w'))
