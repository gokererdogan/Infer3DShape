"""
Inferring 3D Shape from 2D Images

This file contains the PaperClipShape hypothesis class.
This hypothesis assumes objects are made up five tubular connected segments. This type of stimuli was introduced by
Edelman and colleagues and used in research on viewpoint dependency of object recognition.

Created on Feb 24, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
from copy import deepcopy

import Infer3DShape.i3d_hypothesis as hyp
import Infer3DShape.geometry_3d as geom_3d

MIN_SEGMENT_LENGTH = 0.20
MEAN_SEGMENT_LENGTH = 0.60
DEFAULT_SEGMENT_LENGTH_VARIANCE = 0.0001
MAX_XYZ = 1.0


class PaperClipShape(hyp.I3DHypothesis):
    """
    PaperClipShape class. Each shape is assumed to consist of tubular connected segments. We allow the number of joints
    (hence segments) to vary. A minimum and maximum number of joints can be specified. However, if there is no maximum
    number of joints limit, the prior should be changed because a uniform prior cannot be defined over this infinite
    dimensional space.

    Attributes:
        forward_model (VisionForwardModel)
        viewpoint (list)
        params (dict): PaperClipShape requires SEGMENT_LENGTH_VARIANCE parameter. This parameter is only used when creating a
            random object and controls how much segment length varies around the MEAN_SEGMENT_LENGTH.
        joint_positions (list): If not provided, the object is initialized randomly. Each element of the list is a
            3-element numpy.ndarray containing the position of a joint.
        joint_count (int): If joint_positions is not provided, a random shape with joint_count joints is created.
        mid_segment_id (int): Middle segment id. The middle segment is assumed to be aligned with the x-axis. This
            is to ensure that shape representations are unique; othwerwise viewpoint and shape is confounded.
        min_joints (int): Minimum number of joints
        max_joints (int): Maximum number of joints
    """
    def __init__(self, forward_model, viewpoint=None, params=None, joint_positions=None, joint_count=6, min_angle=30.0,
                 max_angle=150.0, mid_segment_id=2, min_joints=2, max_joints=10):
        hyp.I3DHypothesis.__init__(self, forward_model=forward_model, viewpoint=viewpoint, params=params,
                                   primitive_type='TUBE')
        if min_joints < 2:
            min_joints = 2

        if max_joints < min_joints:
            raise ValueError("Maximum number of joints cannot be smaller than minimum number of joints.")

        if max_joints == np.inf:
            raise ValueError("Maximum number of joints has to be finite for the uniform prior. Please define a new "
                             "prior if you want to let it be infinite.")

        self.min_joints = min_joints
        self.max_joints = max_joints

        if mid_segment_id < 0 or mid_segment_id >= self.max_joints:
                raise ValueError("Midsegment id out of bounds.")

        self.mid_segment_id = mid_segment_id

        self.joint_positions = joint_positions
        if self.joint_positions is None:
            # randomly generate a new shape
            self.joint_count = joint_count
            if self.joint_count > self.max_joints or self.joint_count < min_joints:
                raise ValueError("Joint count must be between min_joints and max_joints.")

            if 'SEGMENT_LENGT_VARIANCE' not in self.params:
                self.params['SEGMENT_LENGTH_VARIANCE'] = DEFAULT_SEGMENT_LENGTH_VARIANCE

            length_sd = np.sqrt(self.params['SEGMENT_LENGTH_VARIANCE'])

            self.joint_positions = []

            # add midsegment joints
            midsegment_length = MEAN_SEGMENT_LENGTH + (np.random.randn() * length_sd)
            self.joint_positions.append(np.array((-midsegment_length/2.0, 0.0, 0.0)))
            self.joint_positions.append(np.array((midsegment_length/2.0, 0.0, 0.0)))

            # add left joints
            for i in range(self.mid_segment_id):
                added = False
                while not added:
                    segment_length = MEAN_SEGMENT_LENGTH + (np.random.randn() * length_sd)
                    direction = _get_random_vector_along(self.joint_positions[0] - self.joint_positions[1],
                                                         min_angle=min_angle, max_angle=max_angle)
                    joint_position = self.joint_positions[0] + (segment_length * direction)
                    if np.all(joint_position < MAX_XYZ):
                        self.joint_positions.insert(0, joint_position)
                        added = True

            # add right joints
            for i in range(self.mid_segment_id+2, self.joint_count):
                added = False
                while not added:
                    segment_length = MEAN_SEGMENT_LENGTH + (np.random.randn() * length_sd)
                    direction = _get_random_vector_along(self.joint_positions[-1] - self.joint_positions[-2],
                                                         min_angle=min_angle, max_angle=max_angle)
                    joint_position = self.joint_positions[-1] + (segment_length * direction)
                    if np.all(joint_position < MAX_XYZ):
                        self.joint_positions.append(joint_position)
                        added = True
        else:  # use the initial shape provided in joint_positions
            self.joint_count = len(self.joint_positions)

            if self.mid_segment_id >= self.joint_count-1:
                raise ValueError("Midsegment id out of bounds.")

            if self.joint_count > self.max_joints or self.joint_count < self.min_joints:
                raise ValueError("Number of joints has to be between min_joints and max_joints.")

            # check if midsegment is aligned with the x axis and centered at the origin
            j1 = self.joint_positions[self.mid_segment_id]
            j2 = self.joint_positions[self.mid_segment_id + 1]
            if not np.allclose(j1 + j2, 0.0) or not np.allclose(j1[1:], 0.0) or not np.allclose(j2[1:], 0.0):
                raise ValueError("Middle segment should be centered at the origin and aligned with the x axis.")

            # check segment lengths
            for i in range(self.joint_count-1):
                if self._get_segment_length(i) < MIN_SEGMENT_LENGTH:
                    raise ValueError("Segments cannot be shorter than {0:f}".format(MIN_SEGMENT_LENGTH))

    def _calculate_log_prior(self):
        # define a uniform prior over number of joints
        # this cannot be achieved simply by assigning equal probability to each shape because there are more shapes
        # with n+1 joints than there are with n. So we need to correct for this increase in the number of shapes.
        # It is easy to show that the number of shapes with a given number of joints is proportional to joint_count-1.
        return -np.log(self.joint_count-1)

    def _get_segment_length(self, segment_id):
        if segment_id < 0 or segment_id >= self.joint_count-1:
            raise ValueError('Segment id out of bounds.')

        return np.sqrt(np.sum(np.square(self.joint_positions[segment_id] - self.joint_positions[segment_id+1])))
    
    def _get_joint_angle(self, joint_id):
        if joint_id < 1 or joint_id >= self.joint_count-1:
            raise ValueError('Joint id out out bounds.')

        return geom_3d.angle_between_vectors(self.joint_positions[joint_id-1] - self.joint_positions[joint_id],
                                             self.joint_positions[joint_id+1] - self.joint_positions[joint_id])

    def calculate_moment_of_inertia(self):
        """
        Calculates the moment of inertia of shape.

        We treat each part as an infinitely thin rod.

        Returns:
            float: moment of inertia
        """
        mi = 0.0
        for i in range(self.joint_count-1):
            mi += self._get_segment_moment_of_inertia(i)

        return mi

    def _get_segment_moment_of_inertia(self, segment_id):
        if segment_id < 0 or segment_id >= self.joint_count-1:
            raise ValueError('Segment id out of bounds.')

        xs, ys, _ = self.joint_positions[segment_id]
        xe, ye, _ = self.joint_positions[segment_id+1]
        return (xs**2 + (xs * xe) * xe**2 + ys**2 + (ys * ye) + ye**2) / 3.0

    def change_segment_length(self, segment_id, change_ratio, update_children=False):
        """
        Change segment length by a factor of `change`.

        The direction in which the length of the segment changes depends on where the segment is with
        respect to the midsegment. The joint that is farther away from the midsegment is moved to 
        change the length. For the midsegment, both joints are moved the same amount in opposite 
        directions to ensure that midsegment is centered at the origin.

        Parameters:
            segment_id (int)
            change_ratio (float): ratio of change in length. Has to be greater than -1. Length is multiplied by
                (1 + change)
            update_children: If True children segment positions are updated to keep their lengths the same.
                children refer to the segments that are farther away from the midsegment than the updated
                segment.
        """
        if segment_id < 0 or segment_id >= len(self.joint_positions)-1:
            raise ValueError('Segment id out of bounds.')

        if change_ratio <= -1.0:
            raise ValueError('Length change ratio cannot be less than -1.')

        if segment_id == self.mid_segment_id:
            inc_vector = (change_ratio / 2.0) * (self.joint_positions[self.mid_segment_id] -
                                           self.joint_positions[self.mid_segment_id+1])

            self.move_joint(self.mid_segment_id, inc_vector, update_children=update_children)
        elif segment_id < self.mid_segment_id:
            inc_vector = change_ratio * (self.joint_positions[segment_id] - self.joint_positions[segment_id+1])
            self.move_joint(segment_id, inc_vector, update_children=update_children)
        else:  # segment_id > self.mid_segment_id
            inc_vector = change_ratio * (self.joint_positions[segment_id+1] - self.joint_positions[segment_id])
            self.move_joint(segment_id+1, inc_vector, update_children=update_children)

    def move_joint(self, joint_id, change, update_children=False):
        """
        Move joint by `change`.

        This method moves a given joint (and possibly all of its children) if moving the joint does not make
        any segment smaller than the allowed minimum segment length.

        Parameters:
            joint_id (int)
            change (3-tuple)
            update_children: see ``PaperClipShape.change_segment_length``
        """
        if joint_id < 0 or joint_id >= self.joint_count:
            raise ValueError("Joint id out out bounds.")

        if joint_id == self.mid_segment_id + 1:
            joint_id = self.mid_segment_id
            change = -change

        if joint_id != self.mid_segment_id:
            if self._can_joint_move(joint_id, change, update_children):
                self._move_single_joint(joint_id, change, update_children)
        else:
            if self._can_joint_move(joint_id, change, update_children):
                self._move_single_joint(joint_id, change, update_children)
                self._move_single_joint(joint_id+1, -change, update_children)

    def _can_joint_move(self, joint_id, change, update_children=False):
        """
        Test if joint can be moved. 

        A joint cannot be moved if the any segment becomes shorter than the allowed minimum segment length.

        Parameters:
            joint id (int)
            change (3-tuple)
            update_children: see ``PaperClipShape.change_segment_length``
        """
        if joint_id == self.mid_segment_id + 1:
            joint_id = self.mid_segment_id
            change = -change

        if joint_id == self.mid_segment_id:
            # we can move only in the x direction
            if not np.allclose(change[1:], 0.0):
                return False

            # two joints of the midsegment will move in opposite directions by amount `change`.
            # we need to check the lengths of the midsegment and the two neigboring segments
            new_midsegment_length = self._calculate_new_segment_length(joint_id, 2*change)
            if new_midsegment_length < MIN_SEGMENT_LENGTH:
                return False
            # check the neighboring segments
            if not update_children:
                if joint_id > 0:
                    new_left_length = self._calculate_new_segment_length(joint_id-1, -change)
                    if new_left_length < MIN_SEGMENT_LENGTH:
                        return False
                if joint_id < self.joint_count - 2:
                    new_right_length = self._calculate_new_segment_length(joint_id+1, -change)
                    if new_right_length < MIN_SEGMENT_LENGTH:
                        return False
        elif joint_id < self.mid_segment_id:
            # only one joint on the left side of mid_segment will move.
            # we need to check the lengths of the segments sharing that joint.
            new_segment_length = self._calculate_new_segment_length(joint_id, change)
            if new_segment_length < MIN_SEGMENT_LENGTH:
                return False
            # check neighboring segment
            if not update_children and joint_id > 0:
                new_left_length = self._calculate_new_segment_length(joint_id-1, -change)
                if new_left_length < MIN_SEGMENT_LENGTH:
                    return False
        else: # if joint_id > self.mid_segment_id
            # only one joint on the right side of mid_segment will move.
            # we need to check the lengths of the segments sharing that joint.
            new_segment_length = self._calculate_new_segment_length(joint_id-1, -change)
            if new_segment_length < MIN_SEGMENT_LENGTH:
                return False
            # check neighboring segment
            if not update_children and joint_id < self.joint_count - 1:
                new_right_length = self._calculate_new_segment_length(joint_id, change)
                if new_right_length < MIN_SEGMENT_LENGTH:
                    return False
        return True

    def _calculate_new_segment_length(self, segment_id, change):
        """
        Calculates the new segment length if the left joint of the segment moves by the amount `change`.
        """
        return np.sqrt(np.sum(np.square(self.joint_positions[segment_id] + change -
                                        self.joint_positions[segment_id+1])))

    def _move_single_joint(self, joint_id, change, update_children=False):
        """
        Move joint by amount `change`. If required, it also moves all of its children joints as well.
        """
        self.joint_positions[joint_id] = self.joint_positions[joint_id] + change
        if update_children:
            if joint_id <= self.mid_segment_id:
                for s in range(joint_id):
                    self.joint_positions[s] = self.joint_positions[s] + change
            else:  # joint_id >= self.mid_segment_id+1
                for s in range(joint_id+1, self.joint_count):
                    self.joint_positions[s] = self.joint_positions[s] + change

    def can_rotate_midsegment(self, rot_axis, rot_angle):
        """
        Check if we can rotate the midsegment by rot_angle around rot_axis.

        Parameters:
            rot_axis (numpy.ndarray): Cartesian coordinates of the rotation axis
            rot_angle (float): Rotation amount in degrees

        Returns:
            bool
        """
        old_midsegment = self.joint_positions[self.mid_segment_id]
        new_midsegment = geom_3d.rotate_axis_angle(old_midsegment, rot_axis, rot_angle)

        # we need to check if rotating creates a segment that is too short. Note that this could happen for
        # neighboring segments of the midsegment
        if self.mid_segment_id > 0:
            new_joint_position = geom_3d.rotate_vector_by_vector(self.joint_positions[self.mid_segment_id-1],
                                                                 old_z=new_midsegment, new_z=old_midsegment)

            new_length = np.sqrt(np.sum(np.square(new_joint_position -
                                                  self.joint_positions[self.mid_segment_id])))
            if new_length < MIN_SEGMENT_LENGTH:
                return False

        if self.mid_segment_id < self.joint_count - 2:
            new_joint_position = geom_3d.rotate_vector_by_vector(self.joint_positions[self.mid_segment_id+2],
                                                                 old_z=new_midsegment, new_z=old_midsegment)

            new_length = np.sqrt(np.sum(np.square(new_joint_position -
                                                  self.joint_positions[self.mid_segment_id+1])))
            if new_length < MIN_SEGMENT_LENGTH:
                return False

        return True

    def rotate_midsegment(self, rot_axis, rot_angle):
        """
        Rotates the midsegment.

        Because the midsegment is constrained to point in the x direction, we implement this rotation by rotating the
        other joints.

        Parameters:
            rot_axis (numpy.ndarray): Cartesian coordinates of the rotation axis
            rot_angle (float): Rotation amount in degrees
        """
        if not self.can_rotate_midsegment(rot_axis, rot_angle):
            raise ValueError("Cannot rotate midsegment. Rotation leads to a too short segment.")

        old_midsegment = self.joint_positions[self.mid_segment_id]
        new_midsegment = geom_3d.rotate_axis_angle(old_midsegment, rot_axis, rot_angle)
        # rotate every joint (except the midsegment joints)
        for i in range(self.joint_count):
            if i != self.mid_segment_id and i != self.mid_segment_id + 1:
                # note that we are rotating such that the new midsegment becomes the old one because the joints
                # rotate in the opposite direction to the midsegment
                new_joint_position = geom_3d.rotate_vector_by_vector(self.joint_positions[i], old_z=new_midsegment,
                                                                          new_z=old_midsegment)

                # update joint position
                self.joint_positions[i] = new_joint_position

    def convert_to_positions_sizes(self):
        """
        Returns the positions (start/end points) of segments.
        Used by VisionForwardModel for rendering.

        Returns:
            list of numpy.ndarray: positions
        """
        return self.joint_positions

    def copy(self):
        """
        Returns a (deep) copy of the instance
        """
        # NOTE that we are not copying params. This assumes that
        # parameters do not change from hypothesis to hypothesis.
        self_copy = PaperClipShape(self.forward_model, params=self.params,
                                   viewpoint=deepcopy(self.viewpoint), mid_segment_id=self.mid_segment_id,
                                   joint_positions=[jp.copy() for jp in self.joint_positions],
                                   min_joints=self.min_joints, max_joints=self.max_joints)
        return self_copy

    def __str__(self):
        return "\n".join([str(jp) for jp in self.joint_positions])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, comp):
        if self.joint_count != comp.joint_count:
            return False

        # note that comp can be a mirror image of self or the joints could be in the reverse order
        d1 = [np.sum(np.abs(jp1-jp2)) for jp1, jp2 in zip(self.joint_positions, comp.joint_positions)]
        d2 = [np.sum(np.abs(jp1+jp2)) for jp1, jp2 in zip(self.joint_positions, comp.joint_positions)]
        d3 = [np.sum(np.abs(jp1-jp2)) for jp1, jp2 in zip(self.joint_positions, reversed(comp.joint_positions))]
        d4 = [np.sum(np.abs(jp1+jp2)) for jp1, jp2 in zip(self.joint_positions, reversed(comp.joint_positions))]
        if np.sum(d1) > 0.0 and np.sum(d2) > 0.0 and np.sum(d3) > 0.0 and np.sum(d4) > 0.0:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def can_add_joint(self):
        """
        Check if a new joint can be added.

        This method simply checks if we have reach maximum number of allowed joints.

        Returns:
            bool
        """
        if self.joint_count >= self.max_joints:
            return False

        return True

    def add_joint(self, start_joint_id, new_joint_position):
        """
        Add a new joint to shape. 

        We can add only a new end joint (thereby creating a new segment).

        Parameters:
            start_joint_id (int): Id of the left joint of the new segment. If -1, a new start
                segment is added. If joint_count-1, a new end joint is added.
            new_joint_position (numpy.ndarray)
        """
        if not self.can_add_joint():
            raise ValueError("Already reached maximum number of joints. Cannot add new joint.")

        if start_joint_id != -1 and start_joint_id != self.joint_count-1:
            raise ValueError("Start joint id out out bounds.")

        if new_joint_position is None:
            raise ValueError("New joint position must be provided for end joints.")
        if start_joint_id == -1:
            sid = 0
        else:
            sid = -1
        new_segment_length = np.sqrt(np.sum(np.square(new_joint_position - self.joint_positions[sid])))
        if new_segment_length < MIN_SEGMENT_LENGTH:
            raise ValueError("Cannot add joint because new segment is too short.")
        joint_pos = np.array(new_joint_position)

        self.joint_positions.insert(start_joint_id+1, joint_pos)
        self.joint_count += 1
        if start_joint_id < self.mid_segment_id:
            self.mid_segment_id += 1

    def get_add_joint_choices(self):
        """
        Get the set of joint ids we can use in add_joint method.

        This method is used by ``paperclip_shape_add_remove_joint``.
        """
        choices = [-1, self.joint_count-1]
        return choices

    def can_remove_joint(self):
        """
        Check if we can remove (at least one) joint from the object.

        This method simply checks whether we have more joints than the mininum number of joints allowed.
        """
        if self.joint_count <= self.min_joints:
            return False

        return True

    def remove_joint(self, joint_id):
        """
        Remove a joint. Note that only end joints can be removed and joints of the midsegment cannot be removed.

        Parameters:
            joint_id (int)
        """
        if not self.can_remove_joint():
            raise ValueError("Already reached minimum number of joints. Cannot remove any joint.")

        if joint_id != 0 and joint_id != self.joint_count-1:
            raise ValueError("Joint id out of bounds.")

        if joint_id == self.mid_segment_id or joint_id == self.mid_segment_id + 1:
            raise ValueError("Cannot remove joints of mid segment.")

        self.joint_positions.pop(joint_id)
        self.joint_count -= 1
        if joint_id < self.mid_segment_id:
            self.mid_segment_id -= 1

    def get_remove_joint_choices(self):
        """
        Get the set of joints we can remove.

        This method is used by ``paperclip_shape_add_remove_joint``.
        """
        choices = [0, self.joint_count-1]

        if self.mid_segment_id in choices:
            choices.remove(self.mid_segment_id)

        if self.mid_segment_id+1 in choices:
            choices.remove(self.mid_segment_id+1)

        return choices


def _get_random_vector_along(z_vector, min_angle=30.0, max_angle=180.0):
    """
    Get a random vector that makes more than min_angles and less than max_angles degrees with the `z_vector`.

    This method is used by ``paperclip_shape_add_remove_joint`` move. Note that the angle between the returned vector
    and the -z_vector (NEGATIVE z_vector, not the z_vector) will be in (min_angle, max_angle). If we add such a vector
    to z_vector, the angle between z_vector and the new vector will be in (min_angle, max_angle).
    """
    if max_angle < min_angle:
        raise ValueError("Maximum angle cannot be smaller than minimum angle.")

    max_phi = 180.0 - min_angle
    min_phi = 180.0 - max_angle
    phi = min_phi + (np.random.rand() * (max_phi - min_phi))
    theta = np.random.rand() * 360.0
    coords = geom_3d.spherical_to_cartesian((1.0, theta, phi))
    
    v = geom_3d.rotate_vector_by_vector(coords, old_z=np.array([0., 0., 1.]), new_z=z_vector)
    return v


def paperclip_shape_add_remove_joint(h, params):
    move_choices = []
    if h.can_add_joint():
        move_choices.append('add')
    if h.can_remove_joint():
        move_choices.append('remove')

    if len(move_choices) == 0: # no moves possible
        return h, 1.0, 1.0

    hp = h.copy()

    move = np.random.choice(move_choices)
    if move == 'add':
        # add move
        choices = hp.get_add_joint_choices()

        # pick where to add
        start_joint_id = np.random.choice(choices)

        # pick new joint's position
        if start_joint_id == -1:
            length = MIN_SEGMENT_LENGTH + np.random.rand() * (params['MAX_NEW_SEGMENT_LENGTH'] - MIN_SEGMENT_LENGTH)
            z_vector = hp.joint_positions[0] - hp.joint_positions[1]
            new_joint_pos = hp.joint_positions[0] + (length * _get_random_vector_along(z_vector))
        else:  # start_joint_id == hp.joint_count - 1:
            length = MIN_SEGMENT_LENGTH + np.random.rand() * (params['MAX_NEW_SEGMENT_LENGTH'] - MIN_SEGMENT_LENGTH)
            z_vector = hp.joint_positions[-1] - hp.joint_positions[-2]
            new_joint_pos = hp.joint_positions[-1] + (length * _get_random_vector_along(z_vector))

        # add joint
        hp.add_joint(start_joint_id, new_joint_pos)

        q_hp_h = 0.5 * (1.0 / len(choices))
        # if add is the only move possible
        if len(move_choices) == 1:
            q_hp_h = 1.0 * (1.0 / len(choices))

        # q(h|hp)
        remove_choice_count = len(hp.get_remove_joint_choices())
        q_h_hp = 0.5 * (1.0 / remove_choice_count)
        #  if remove is the only possible reverse move
        if not hp.can_add_joint():
            q_h_hp = 1.0 * (1.0 / remove_choice_count)
    else:  # move == 'remove'
        choices = hp.get_remove_joint_choices()
        remove_id = np.random.choice(choices)
        hp.remove_joint(remove_id)

        q_hp_h = 0.5 * (1.0 / len(choices))
        # if remove move is the only possible move
        if len(move_choices) == 1:
            q_hp_h = 1.0 * (1.0 / len(choices))

        add_choice_count = len(hp.get_add_joint_choices())
        q_h_hp = 0.5 * (1.0 / add_choice_count)
        if not hp.can_remove_joint():
            q_h_hp = 1.0 * (1.0 / add_choice_count)

    return hp, q_hp_h, q_h_hp


def paperclip_shape_move_joint(h, params):
    hp = h.copy()

    choices = range(len(hp.joint_positions))
    joint_id = np.random.choice(choices)
    move_amount = (np.random.randn(3) * np.sqrt(params['MOVE_JOINT_VARIANCE']))
    hp.move_joint(joint_id, move_amount, update_children=False)

    return hp, 1.0, 1.0


def paperclip_shape_move_branch(h, params):
    hp = h.copy()

    choices = range(len(hp.joint_positions))
    joint_id = np.random.choice(choices)
    move_amount = (np.random.randn(3) * np.sqrt(params['MOVE_JOINT_VARIANCE']))
    hp.move_joint(joint_id, move_amount, update_children=True)

    return hp, 1.0, 1.0


def paperclip_shape_change_segment_length(h, params):
    if np.abs(params['MAX_SEGMENT_LENGTH_CHANGE']) > 1.0:
        raise ValueError("Maximum segment length change ratio must be less than 1.0.")

    hp = h.copy()

    segment_id = np.random.randint(0, len(hp.joint_positions)-1)
    change = (np.random.rand() - 0.5) * params['MAX_SEGMENT_LENGTH_CHANGE'] * 2.0
    hp.change_segment_length(segment_id, change, update_children=False)

    return hp, 1.0, 1.0


def paperclip_shape_change_branch_length(h, params):
    if np.abs(params['MAX_SEGMENT_LENGTH_CHANGE']) > 1.0:
        raise ValueError("Maximum segment length change ratio must be less than 1.0.")

    hp = h.copy()

    segment_id = np.random.randint(0, len(hp.joint_positions)-1)
    change = (np.random.rand() - 0.5) * params['MAX_SEGMENT_LENGTH_CHANGE'] * 2.0
    hp.change_segment_length(segment_id, change, update_children=True)

    return hp, 1.0, 1.0


def paperclip_shape_rotate_midsegment(h, params):
    """
    This move rotates the midsegment of h, keeping the other segments fixed.

    Because we constrain midsegment to be aligned with the x axis, this rotation is implemented by rotating the other
    segments. Crucially, we want it to look like only the midsegment is rotated; so we rotate the viewpoint to do that.
    """
    hp = h.copy()

    # if the object has only a midsegment
    if hp.joint_count == 2:
        return hp, 1.0, 1.0

    # get random rotation axis and angle
    theta = np.random.rand() * 360.0 - 180.0
    phi = np.random.rand() * 180.0
    rotation_axis = geom_3d.spherical_to_cartesian((1.0, theta, phi))

    kappa = 1 / (params['ROTATE_MIDSEGMENT_VARIANCE'] * np.pi**2 / 180**2)
    rotation_angle = np.random.vonmises(0.0, kappa) * 180.0 / np.pi

    # rotate midsegment
    try:  # we might not be able to rotate midsegment by rotation_angle around rotation_axis
        hp.rotate_midsegment(rotation_axis, rotation_angle)
    except ValueError:
        return hp, 1.0, 1.0

    # rotate each viewpoint to correct for the rotation of the joints (we want only the midsegment to change how it
    # looks)
    for i in range(len(hp.viewpoint)):
        vp_cartesian = geom_3d.spherical_to_cartesian(hp.viewpoint[i])
        new_vp_cartesian = geom_3d.rotate_axis_angle(vp_cartesian, rotation_axis, -rotation_angle)
        hp.viewpoint[i] = geom_3d.cartesian_to_spherical(new_vp_cartesian)

    return hp, 1.0, 1.0


if __name__ == "__main__":
    import vision_forward_model as vfm
    import mcmclib.proposal
    import i3d_proposal

    fwm = vfm.VisionForwardModel(render_size=(200, 200), offscreen_rendering=False, custom_lighting=False)
    h = PaperClipShape(forward_model=fwm, viewpoint=[np.array((np.sqrt(8.0), 0.0, 30.0))], min_joints=2, max_joints=10,
                           params={'LL_VARIANCE': 0.0001, 'MAX_PIXEL_VALUE': 255.0, 'SEGMENT_LENGTH_VARIANCE': 0.0001})

    moves = {'paperclip_move_joints': paperclip_shape_move_joint,
             'paperclip_move_branch': paperclip_shape_move_branch,
             'paperclip_change_segment_length': paperclip_shape_change_segment_length,
             'paperclip_change_branch_length': paperclip_shape_change_branch_length,
             'paperclip_add_remove_joint': paperclip_shape_add_remove_joint,
             'paperclip_rotate_midsegment': paperclip_shape_rotate_midsegment,
             'change_viewpoint': i3d_proposal.change_viewpoint}

    params = {'MOVE_JOINT_VARIANCE': 0.005,
              'MAX_NEW_SEGMENT_LENGTH': 0.6,
              'MAX_SEGMENT_LENGTH_CHANGE': 0.6,
              'ROTATE_MIDSEGMENT_VARIANCE': 60.0,
              'CHANGE_VIEWPOINT_VARIANCE': 30.0}

    proposal = mcmclib.proposal.RandomMixtureProposal(moves, params)

    data = np.load('data/test6_single_view.npy')

    # choose sampler
    thinning_period = 500
    sampler_class = 'mh'
    if sampler_class == 'mh':
        import mcmclib.mh_sampler
        sampler = mcmclib.mh_sampler.MHSampler(h, data, proposal, burn_in=1000, sample_count=10, best_sample_count=6,
                                               thinning_period=thinning_period, report_period=thinning_period)
    elif sampler_class == 'pt':
        from mcmclib.parallel_tempering_sampler import ParallelTemperingSampler
        sampler = ParallelTemperingSampler(initial_hs=[h, h, h], data=data, proposals=[proposal, proposal, proposal],
                                           temperatures=[3.0, 1.5, 1.0], burn_in=1000, sample_count=10,
                                           best_sample_count=10, thinning_period=int(thinning_period / 3.0),
                                           report_period=int(thinning_period / 3.0))
    else:
        raise ValueError('Unknown sampler class')

    run = sampler.sample()
