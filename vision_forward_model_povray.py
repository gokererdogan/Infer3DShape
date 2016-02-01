"""
Inferring 3D Shape from 2D Images

This file contains an alternative vision forward model that
uses POV-ray to render hypotheses to 2D images. The advantage
in using POV-Ray is we do not need to rely on X-server and
OpenGL for rendering; that allows us to run the forward model
through SSH etc. with no X-server.

Created on Oct 13, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import vapory # https://github.com/Zulko/vapory
import numpy as np
import scipy.misc
from gmllib.helpers import rgb2gray

# canonical view +- 45 degrees
DEFAULT_CAMERA_POS = [(3.0, 3.0, -3.0), (4.242, 3.0, 0.0), (0.0, 3.0, -4.242), (3.0, 3.0, 3.0), (-3.0, 3.0, -3.0)]
DEFAULT_RENDER_SIZE = (300, 300)

class VisionForwardModelPOVRay:
    """
    Vision forward model for Shape class (hypothesis.py) 
    Creates 3D scene according to given shape representation 
    and uses POVRay via vapory to render 3D scene to 2D image
    Each part is assumed to be a rectangular prism.
    Forward model expects a Shape (from hypothesis.py)
    instance which contains the position and size of each
    part
    """
    def __init__(self, render_size=DEFAULT_RENDER_SIZE, camera_pos=DEFAULT_CAMERA_POS):
        """
        Initializes VTK objects for rendering.
        """
        self.render_size = render_size
        self.camera_pos = camera_pos

        self.camera_view_count = len(self.camera_pos)
        
        self.cameras = []
        for pos in self.camera_pos:
            self.cameras.append(vapory.Camera('location', pos, 'look_at', [0.0, 0.0, 0.0]))

        # lighting
        self.light = vapory.LightSource([2.0, 3.0, -2.5], 'color', [1.0, 1.0, 1.0], 'shadowless')

    def render(self, shape):
        """
        Construct the 3D object from Shape instance and render it.
        Returns numpy array with size number of viewpoints x self.render_size
        """
        objects = self._build_scene(shape)
        w = self.render_size[0]
        h = self.render_size[1]
        img_arr = np.zeros((self.camera_view_count, w, h))
        for i, camera in enumerate(self.cameras):
            img_arr[i, :, :] = self._render_scene(camera, objects)
        return img_arr
    
    def _render_scene(self, camera, objects):
        """
        Renders the window to 2D grayscale image
        Called from render function for each viewpoint
        """
        scene = vapory.Scene(camera=camera, objects=objects)
        img = scene.render(width=self.render_size[0], height=self.render_size[1], quality=3)
        img = rgb2gray(img)

        return img
    
    def _build_scene(self, shape):
        """
        Creates each object and appends them to the objects list
        """

        objects = []
        # add parts to scene
        positions, sizes = shape.convert_to_positions_sizes()
        for position, size in zip(positions, sizes):
            lb = position - (size / 2.0)
            ub = position + (size / 2.0)
            objects.append(vapory.Box(lb, ub, vapory.Texture(vapory.Pigment('color', [1.0, 1.0, 1.0]))))

        # add light
        objects.append(self.light)
        return objects

    def save_render(self, filename, shape):
        """
        Save rendered image to disk.
        """
        fp = filename.split('.')
        fn = ".".join(fp[0:-1])
        ext = fp[-1]
        img = self.render(shape)
        # make sure that scipy does not normalize the image
        for i in range(self.camera_view_count):
            simg = scipy.misc.toimage(img[i], cmin=0, cmax=255)
            simg.save("{0:s}_{1:d}.{2:s}".format(fn, i, ext))

        
if __name__ == '__main__':
    forward_model = VisionForwardModelPOVRay()

    import bdaooss_shape as bdaooss
    s = bdaooss.BDAoOSSShapeMaxD(forward_model=forward_model)

    forward_model.save_render('r.png', s)
