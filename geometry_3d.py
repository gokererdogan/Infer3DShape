"""
Inferring 3D Shape from 2D Images

This file contains miscellaneous helper functions dealing with 3D geometry.

Created on Mar 3, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np


def cartesian_to_spherical(coords):
    """
    Convert cartesian 3D coordinates to spherical coordinates.
        Spherical coordinates are (r, theta, phi)
        r is the radial distance, theta is the azimuthal angle, and phi is the polar angle.
        See https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical_2.svg
        We use the conventions in the mathematical community; theta is the angle in x-y plane.
        theta is in [-180, 180], phi is in [0, 180]

    Parameters:
        coords (tuple or numpy.ndarray): (x, y, z)

    Returns:
        (numpy.ndarray): spherical coordinates (r, theta, phi) of the input
    """
    x, y, z = coords
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) * 180.0 / np.pi
    phi = np.arctan2(np.sqrt(x**2 + y**2), z) * 180.0 / np.pi
    return np.array([r, theta, phi])


def spherical_to_cartesian(coords):
    """
    Convert spherical coordinates to cartesian coordinates.
        Spherical coordinates are given by (r, theta, phi). See `cartesian_to_spherical` for more info.

    Parameters:
        coords (tuple or numpy.ndarray): spherical coordinates (r, theta, phi)

    Returns:
        (numpy.ndarray): Cartesian (x, y, z) coordinates
    """
    r, theta, phi = coords
    theta = ((theta + 180.0) % 360.0) - 180.0
    theta = theta / 180.0 * np.pi
    phi = (phi % 360.0)
    phi = phi / 180.0 * np.pi
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def rotate_vector_by_vector(v, old_z, new_z):
    """
    Rotate vector v by the rotation aligning oldz with newz.
        Finds the rotation aligning oldz to newz and applies it to v. This can also be viewed as rotating the coordinate
        system with z axis oldz so that z axis points in the direction of newz in the new coordinate system.
        All vectors are given in Cartesian coordinates.

    Parameters:
        v (numpy.ndarray): Cartesian coordinates of the point to rotate
        old_z (numpy.ndarray): Old z direction
        new_z (numpy.ndarray): New z direction

    Returns:
        (numpy.ndarray): Rotated vector (x, y, z)
    """
    axis, angle = vectors_to_axis_angle(old_z, new_z)
    return rotate_axis_angle(v, axis, angle)


def rotate_axis_angle(v, axis, angle):
    """
    Rotate vector v around axis by angle.

    Parameters:
        v (numpy.ndarray): Cartesian coordinates of the vector to rotate.
        axis (numpy.ndarray): Cartesian coordinates of the rotation axis
        angle (float): Rotation amount, given in degrees.

    Returns:
        (numpy.ndarray): Rotated vector (x, y, z)
    """
    if np.allclose(axis, 0.0) or np.allclose(v, 0.0):
        return v

    axis /= np.linalg.norm(axis)
    angle = angle / 180.0 * np.pi

    # use Rodrigues' formula
    new_v = (v * np.cos(angle)) + (np.cross(axis, v) * np.sin(angle)) + (axis * np.dot(axis, v) * (1 - np.cos(angle)))

    return new_v


def vectors_to_axis_angle(v1, v2):
    """
    Calculate the axis of rotation and the rotation amount to rotate v1 onto v2.

    Parameters:
        v1 (numpy.ndarray): Vector to rotate in Cartesian coordinates
        v2 (numpy.ndarray): Vector to rotate to in Cartesian coordinates

    Returns:
        (numpy.ndarray): Axis of rotation in Cartesian coordinates
        (float): Rotation angle in degrees
    """
    axis = np.cross(v1, v2)

    if np.allclose(axis, 0):
        axis = np.array([0., 0., 1.])

    axis /= np.linalg.norm(axis)
    angle = angle_between_vectors(v1, v2)
    return axis, angle


def angle_between_vectors(v1, v2):
    """
    Calculate the angle between vectors v1 and v2.

    Parameters:
        v1 (numpy.ndarray): Vector 1 in Cartesian coordinates
        v2 (numpy.ndarray): Vector 2 in Cartesian coordinates

    Returns:
        (float): Angle between vectors in degrees
    """
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    if np.isclose(m1, 0.0) or np.isclose(m2, 0.0):
        return 0.0

    return np.arccos(np.dot(v1, v2) / (m1 * m2)) * 180.0 / np.pi


