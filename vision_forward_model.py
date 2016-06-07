"""
Inferring 3D Shape from 2D Images

This file contains the vision forward model that renders  
hypotheses to 2D images.

Created on Aug 27, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import vtk

from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import scipy.misc
from gmllib.helpers import rgb2gray
import geometry_3d as geom3d

DEFAULT_CAMERA_DISTANCE = np.sqrt(8.0)
# camera position is given in spherical coordinates
# canonical view +- 45 degrees
DEFAULT_CAMERA_POS = [(DEFAULT_CAMERA_DISTANCE, -45.0, 45.0),
                      (DEFAULT_CAMERA_DISTANCE, 0.0, 45.0),
                      (DEFAULT_CAMERA_DISTANCE, 45.0, 45.0)]

DEFAULT_RENDER_SIZE = (200, 200)

# tube radius
TUBE_RADIUS = 0.05


class VisionForwardModel:
    """
    Vision forward model for Shape class (hypothesis.py) 
    Creates 3D scene according to given shape representation 
    and uses VTK to render 3D scene to 2D image
    Each part is assumed to be a rectangular prism.
    Forward model expects an I3DHypothesis instance
    that implements the method convert_to_positions_sizes
    """
    def __init__(self, render_size=DEFAULT_RENDER_SIZE, camera_pos=DEFAULT_CAMERA_POS,
                 offscreen_rendering=True, custom_lighting=True):
        """
        Initializes VTK objects for rendering.
        """
        self.render_size = render_size
        self.camera_pos = camera_pos

        # vtk objects for rendering
        self.vtkrenderer = vtk.vtkRenderer()

        self.camera_view_count = len(self.camera_pos)
        
        self.vtkcamera = vtk.vtkCamera()
        self._setup_camera(self.camera_pos[0])
        # this is the view angle we used for rendering objects in blender.
        self.vtkcamera.SetViewAngle(35.0)

        # if custom_lighting is ON, we use our own lights in the scene. default lighting of vtk is not very good.
        # this is the default (used for our BDAoOSS model and CogSci16 paper).
        # we illuminate the upper hemisphere.
        self.custom_lighting = custom_lighting
        if self.custom_lighting:
            self.light1 = vtk.vtkLight()
            self.light1.SetIntensity(.3)
            self.light1.SetPosition(8, -12, 10)
            self.light1.SetDiffuseColor(1.0, 1.0, 1.0)
            self.light2 = vtk.vtkLight()
            self.light2.SetIntensity(.3)
            self.light2.SetPosition(-12, -10, 8)
            self.light2.SetDiffuseColor(1.0, 1.0, 1.0)
            self.light3 = vtk.vtkLight()
            self.light3.SetIntensity(.3)
            self.light3.SetPosition(10, 10, 12)
            self.light3.SetDiffuseColor(1.0, 1.0, 1.0)
            self.light4 = vtk.vtkLight()
            self.light4.SetIntensity(.3)
            self.light4.SetPosition(-10, 8, 10)
            self.light4.SetDiffuseColor(1.0, 1.0, 1.0)

            self.vtkrenderer.AddLight(self.light1)
            self.vtkrenderer.AddLight(self.light2)
            self.vtkrenderer.AddLight(self.light3)
            self.vtkrenderer.AddLight(self.light4)

        self.vtkrenderer.SetBackground(0.0, 0.0, 0.0)  # Background color

        self.vtkrender_window = vtk.vtkRenderWindow()
        self.vtkrender_window.AddRenderer(self.vtkrenderer)
        self.vtkrender_window.SetSize(self.render_size)
        self.vtkrender_window_interactor = vtk.vtkRenderWindowInteractor()
        self.vtkrender_window_interactor.SetRenderWindow(self.vtkrender_window)

        # turn on off-screen rendering. Note that rendering will be much faster this way
        # HOWEVER, you won't be able to use view etc. methods to look at and interact with
        # the object. You must render the object and use matplotlib etc. to view the
        # rendered image.
        self.offscreen_rendering = offscreen_rendering
        if self.offscreen_rendering:
            self.vtkrender_window.SetOffScreenRendering(1)

        # these below lines are here with the hope of fixing a problem with off-screen rendering.
        # the quality of the offscreen render seems inferior.
        # i'm not sure why this is.
        # 1) it could be because there is a bug and anti-aliasing and multisampling is disabled
        # for offscreen rendering. see the below link (though that applies to vtk6.0)
        # http://public.kitware.com/pipermail/vtk-developers/2015-April/031741.html
        # 2) it might be that offscreen rendering uses software rendering, no hardware
        # acceleration. I don't know how one can check that.
        # still, the quality of the renders are good enough for our purposes
        self.vtkrender_window.SetLineSmoothing(1)
        self.vtkrender_window.SetMultiSamples(8)

        # vtk objects for reading, and rendering object parts
        self.cube_source = vtk.vtkCubeSource()
        self.cube_output = self.cube_source.GetOutput()
        self.cube_mapper = vtk.vtkPolyDataMapper()
        self.cube_mapper.SetInput(self.cube_output)

    def _reset_camera(self):
        """
        Reset camera to its original position
        """
        self._setup_camera(self.camera_pos[0])

    def _setup_camera(self, camera_pos):
        """
        Set the camera position, view up and roll

            camera_pos (3-tuple): Camera position in spherical coordinates
        """
        camera_pos_cartesian = geom3d.spherical_to_cartesian(camera_pos)
        self.vtkcamera.SetPosition(camera_pos_cartesian)
        # calculate camera up
        # if viewpoint radius is negative, we invert the view, i.e., rotate camera upside down
        camera_up = self._calculate_camera_up(camera_pos)
        self.vtkcamera.SetViewUp(camera_up)

    @staticmethod
    def _calculate_camera_up(camera_pos):
        """Calculate camera up direction from camera position.

        Parameters:
            camera_pos (tuple): spherical coordinates of camera position

        Returns:
            (tuple): camera up vector in cartesian coordinates

        """
        # when camera position is (r, theta=0, phi=0)=(0, 0, z), camera up is (-1, 0, 0)
        x, y, z = -1.0, 0.0, 0.0
        # get the spherical coordinates of camera pos and rotate the camera up vector
        _, theta, phi = camera_pos
        phi *= (np.pi / 180.0)
        theta *= (np.pi / 180.0)

        # rotate by phi wrt to y
        xr = (np.cos(phi) * x) + (np.sin(phi) * z)
        yr = y
        zr = (-np.sin(phi) * x) + (np.cos(phi) * z)

        # rotate by theta wrt to z
        x = (np.cos(theta) * xr) - (np.sin(theta) * yr)
        y = (np.sin(theta) * xr) + (np.cos(theta) * yr)
        z = zr

        return x, y, z
 
    def render(self, shape, rgb=False):
        """
        Construct the 3D object from Shape instance and render it.
        Returns numpy array with size number of viewpoints x self.render_size
        If viewpoints are not defined in shape, it uses default viewpoints specified in this file.

        Parameters:
            shape (I3DHypothesis): shape to render. should contain primitive_type attribute and
                convert_to_positions_sizes method.
            rgb (bool): return a grayscale or RGB image

        Returns:
            (numpy.ndarray): rendered image of the object from the specified viewpoints.
                an array of viewpoints x self.render_size
        """
        self._build_scene(shape)
        w = self.render_size[0]
        h = self.render_size[1]

        camera_positions = self.camera_pos
        if shape.viewpoint is not None:
            camera_positions = shape.viewpoint

        viewpoint_count = len(camera_positions)
        if rgb:
            img_arr = np.zeros((viewpoint_count, 3, w, h), dtype=np.uint8)
        else:
            img_arr = np.zeros((viewpoint_count, w, h))

        for i, camera_pos in enumerate(camera_positions):
            self._setup_camera(camera_pos)
            img_arr[i] = self._render_window_to2D(rgb)

        return img_arr
    
    def _render_window_to2D(self, rgb=False):
        """
        Renders the window to 2D grayscale image
        Called from render function for each viewpoint
        """
        self.vtkrender_window.Render()
        self.vtkwin_im = vtk.vtkWindowToImageFilter()
        self.vtkwin_im.SetInput(self.vtkrender_window)
        self.vtkwin_im.Update()

        vtk_image = self.vtkwin_im.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        if rgb:
            arr = arr.transpose([2, 1, 0])
        else:
            arr = rgb2gray(arr)

        return arr
    
    def _build_scene(self, shape):
        """
        Places each part of the shape into the scene
        """
        # clear scene
        self.vtkrenderer.RemoveAllViewProps()
        self.vtkrenderer.Clear()
        # add parts to scene, use the build_scene function for the primitive type of shape
        if shape.primitive_type == 'CUBE':
            self._build_scene_cube(shape)
        elif shape.primitive_type == 'TUBE':
            self._build_scene_tube(shape)
        else:
            raise ValueError("Unknown primitive type.")

        self._reset_camera()
        self.vtkrenderer.SetActiveCamera(self.vtkcamera)

    def _build_scene_cube(self, shape):
        positions, sizes = shape.convert_to_positions_sizes()
        for position, size in zip(positions, sizes):
            actor = vtk.vtkActor()
            actor.SetMapper(self.cube_mapper)
            actor.SetPosition(position)
            actor.SetScale(size)
            self.vtkrenderer.AddActor(actor)

    def _build_scene_tube(self, shape):
        positions = shape.convert_to_positions_sizes()
        joint_count = len(positions)
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(joint_count)
        for j in range(joint_count):
            pts.InsertPoint(j, positions[j])
            lines.InsertCellPoint(j)
        td = vtk.vtkPolyData()
        td.SetPoints(pts)
        td.SetLines(lines)
        tf = vtk.vtkTubeFilter()
        tf.SetInput(td)
        tf.SetRadius(TUBE_RADIUS)
        # tf.SetVaryRadiusToVaryRadiusOff()
        tf.SetCapping(1)
        tf.SetNumberOfSides(50)
        tf.Update()
        tm = vtk.vtkPolyDataMapper()
        tm.SetInput(tf.GetOutput())
        ta = vtk.vtkActor()
        ta.SetMapper(tm)
        #ta.GetProperty().SetDiffuse(0.8)
        ta.GetProperty().SetAmbient(0.25)
        self.vtkrenderer.AddActor(ta)

    def convert_world_to_display(self, viewpoint, x, y, z):
        """Converts world coordinates x, y, z to display coordinates.

        Used in 2D affine alignment model to get shape feature coordinates
        in image.
        """
        self.vtkrenderer.SetActiveCamera(self.vtkcamera)
        self._setup_camera(viewpoint)

        vtkcoordinate = vtk.vtkCoordinate()
        vtkcoordinate.SetCoordinateSystemToWorld()
        vtkcoordinate.SetValue(x, y, z)
        # x and y are flipped in render method.
        y, x = vtkcoordinate.GetComputedDisplayValue(self.vtkrenderer)
        return x, y

    def _view(self, shape):
        """
        Views object in window
        Used for development and testing purposes
        """
        self._build_scene(shape)

        if shape.viewpoint is not None:
            camera_pos = shape.viewpoint[0]
            self._setup_camera(camera_pos)

        self.vtkrender_window.Render()
        self.vtkrender_window_interactor.Start()

    def save_render(self, filename, shape):
        """
        Save rendered image to disk. Saves one image for each viewpoint.

        Parameters:
            filename (str): save filename with extension.
            shape (I3DHypothesis): shape to render

        Returns:
            None
        """
        fp = filename.split('.')
        fn = ".".join(fp[0:-1])
        ext = fp[-1]
        img = self.render(shape)
        # make sure that scipy does not normalize the image
        for i in range(img.shape[0]):
            simg = scipy.misc.toimage(np.flipud(img[i]), cmin=0, cmax=255)
            simg.save("{0:s}_{1:d}.{2:s}".format(fn, i, ext))

if __name__ == '__main__':
    forward_model = VisionForwardModel(render_size=(200, 300))

    import shape as hyp
    s = hyp.Shape(forward_model)
    r = forward_model.render(s)
    forward_model.save_render('r.png', s)
