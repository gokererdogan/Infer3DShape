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

# canonical view +- 45 degrees
DEFAULT_CAMERA_POS = [(1.5, -1.5, 1.5), (2.121, 0.0, 1.5), (0.0, -2.121, 1.5), (1.5, 1.5, 1.5), (-1.5, -1.5, 1.5)]
DEFAULT_RENDER_SIZE = (100, 100)
DEFAULT_CAMERA_UP = (0, 0, 1)

class VisionForwardModel:
    """
    Vision forward model for Shape class (hypothesis.py) 
    Creates 3D scene according to given shape representation 
    and uses VTK to render 3D scene to 2D image
    Each part is assumed to be a rectangular prism.
    Forward model expects a Shape (from hypothesis.py)
    instance which contains the position and size of each
    part
    """
    def __init__(self, render_size=DEFAULT_RENDER_SIZE, camera_pos=DEFAULT_CAMERA_POS, camera_up=DEFAULT_CAMERA_UP):
        """
        Initializes VTK objects for rendering.
        """
        self.render_size = render_size
        self.camera_pos = camera_pos
        self.camera_up = camera_up

        # vtk objects for rendering
        self.vtkrenderer = vtk.vtkRenderer()

        self.camera_view_count = len(self.camera_pos)
        
        self.vtkcamera = vtk.vtkCamera()
        self.vtkcamera.SetPosition(self.camera_pos[0])
        self.vtkcamera.SetFocalPoint(0, 0, 0)
        self.vtkcamera.SetViewUp(self.camera_up)

        # lighting
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

        self.vtkrenderer.SetBackground(0.0, 0.0, 0.0) # Background color

        self.vtkrender_window = vtk.vtkRenderWindow()
        self.vtkrender_window.AddRenderer(self.vtkrenderer)
        self.vtkrender_window.SetSize(self.render_size)
        self.vtkrender_window_interactor = vtk.vtkRenderWindowInteractor()
        self.vtkrender_window_interactor.SetRenderWindow(self.vtkrender_window)

        # turn on off-screen rendering. Note that rendering will be much faster this way
        # HOWEVER, you won't be able to use view etc. methods to look at and interact with
        # the object. You must render the object and use matplotlib etc. to view the
        # rendered image.
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
        self.part_source = vtk.vtkCubeSource()
        self.part_output = self.part_source.GetOutput()
        self.part_mapper = vtk.vtkPolyDataMapper()
        self.part_mapper.SetInput(self.part_output)

        # exporters
        self.vtkvrml_exporter = vtk.vtkVRMLExporter()
        self.vtkobj_exporter = vtk.vtkOBJExporter()
        self.stl_writer = vtk.vtkSTLWriter()

    def _reset_camera(self):
        """
        Reset camera to its original position
        """
        self.vtkcamera.SetPosition(self.camera_pos[0])
        self.vtkcamera.SetFocalPoint(0, 0, 0)
        self.vtkcamera.SetViewUp(self.camera_up)
 
    def render(self, shape):
        """
        Construct the 3D object from Shape instance and render it.
        Returns numpy array with size number of viewpoints x self.render_size
        """
        self._build_scene(shape)
        w = self.render_size[0]
        h = self.render_size[1]
        # if shape has viewpoint defined, use that
        camera_pos = self.camera_pos
        if shape.viewpoint is not None:
            camera_pos = shape.viewpoint
        img_arr = np.zeros((len(camera_pos), w, h))
        for i, pos in enumerate(camera_pos):
            self.vtkcamera.SetPosition(pos)
            img_arr[i, :, :] = self._render_window_to2D()
        return img_arr
    
    def _render_window_to2D(self):
        """
        Renders the window to 2D grayscale image
        Called from render function for each viewpoint
        """
        self.vtkrender_window.Render()
        self.vtkwin_im = vtk.vtkWindowToImageFilter()
        self.vtkwin_im.SetInput(self.vtkrender_window)
        self.vtkwin_im.Update()

        vtk_image = self.vtkwin_im.GetOutput()
        height, width, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
        arr = rgb2gray(arr)

        return arr
    
    def _build_scene(self, shape):
        """
        Places each part of the shape into the scene
        Returns vtkRenderer
        """
        # clear scene
        self.vtkrenderer.RemoveAllViewProps()
        self.vtkrenderer.Clear()
        # add parts to scene
        positions, sizes = shape.convert_to_positions_sizes()
        for position, size in zip(positions, sizes):
            actor = vtk.vtkActor()
            actor.SetMapper(self.part_mapper)
            actor.SetPosition(position)
            actor.SetScale(size)
            self.vtkrenderer.AddActor(actor)
        self._reset_camera()
        self.vtkrenderer.SetActiveCamera(self.vtkcamera)
                
    def _view(self, shape):
        """
        Views object in window
        Used for development and testing purposes
        """
        self._build_scene(shape)
        self.vtkrender_window.Render()
        self.vtkrender_window_interactor.Start()

    def _save_wrl(self, filename, shape):
        """
        Save object to wrl file.
        """
        self._build_scene(shape)
        self.vtkrender_window.Render()
        self.vtkvrml_exporter.SetInput(self.vtkrender_window)
        self.vtkvrml_exporter.SetFileName(filename)
        self.vtkvrml_exporter.Write()
    
    def _save_obj(self, filename, shape):
        """
        Save object to obj file.
        """
        self._build_scene(shape)
        self.vtkrender_window.Render()
        self.vtkobj_exporter.SetInput(self.vtkrender_window)
        self.vtkobj_exporter.SetFilePrefix(filename)
        self.vtkobj_exporter.Write()

    def _save_stl(self, filename, shape):
        """
        Save object to stl file.
        """
        return NotImplementedError()
        # TO-DO
        # we can't save the whole scene to a single STL file,
        # we need to figure out how to get around that.
        
    def save_render(self, filename, shape):
        """
        Save rendered image to disk.
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
    forward_model = VisionForwardModel()

    import shape as hyp
    s = hyp.Shape(forward_model)

    forward_model.save_render('r.png', s)
