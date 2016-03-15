"""
Created on Nov 17, 2015

Goker Erdogan
gokererdogan@gmail.com

VTK development code
Create two different cubes with the same projection
"""

# 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

# for the following scene setup with a camera pointing towards
# origin at (0, 0, 5) and a single cube in the scene of unit
# width, height, depth at the origin, we can find infinitely
# many cubes that will have the same projection.
# as long as the z coordinate of cube is -5*(depth-1), we will
# get the same projection.

w = (np.random.rand() * 5.0) + 0.1
z = -5 * (w - 1)
p1Xw = w
p1Yw = w
p1Zw = w
p1x = 0
p1y = 0
p1z = z

part1 = vtk.vtkCubeSource()
part1.SetXLength(p1Xw)
part1.SetYLength(p1Yw)
part1.SetZLength(p1Zw)
part1Output = part1.GetOutput()

# Create a mapper and actor
part1Mapper = vtk.vtkPolyDataMapper()
part1Mapper.SetInput(part1Output)
part1Actor = vtk.vtkActor()
part1Actor.SetMapper(part1Mapper)
part1Actor.SetPosition(p1x, p1y, p1z)
part1Actor.SetScale(1, 1, 1)
part1Actor.RotateZ(145.0)
part1Actor.RotateY(475.0)
part1Actor.RotateX(459.0)

camera = vtk.vtkCamera()
camera.SetPosition(0, 0, 5)
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 1, 0)

# x, y, z lines
# create source
xl = vtk.vtkLineSource()
xl.SetPoint1(-10, 0, 0)
xl.SetPoint2(10, 0, 0)
yl = vtk.vtkLineSource()
yl.SetPoint1(0, -10, 0)
yl.SetPoint2(0, 10, 0)
zl = vtk.vtkLineSource()
zl.SetPoint1(0, 0, -10)
zl.SetPoint2(0, 0, 10)
 
# mapper
mapperx = vtk.vtkPolyDataMapper()
mapperx.SetInput(xl.GetOutput())
mappery = vtk.vtkPolyDataMapper()
mappery.SetInput(yl.GetOutput())
mapperz = vtk.vtkPolyDataMapper()
mapperz.SetInput(zl.GetOutput())

# actor
actorx = vtk.vtkActor()
actorx.SetMapper(mapperx)
actory = vtk.vtkActor()
actory.SetMapper(mappery)
actorz = vtk.vtkActor()
actorz.SetMapper(mapperz)

# color actor
actorx.GetProperty().SetColor(1, 0, 0)
actory.GetProperty().SetColor(0, 1, 0)
actorz.GetProperty().SetColor(0, 1, 1)


pts = vtk.vtkPoints()
pts.InsertPoint(0, -0.6, 0.5, 0.0)
pts.InsertPoint(1, -0.2, 0.0, 0.0)
pts.InsertPoint(2, 0.5, 0.0, 0.0)

lines = vtk.vtkCellArray()
lines.InsertNextCell(3)
lines.InsertCellPoint(0)
lines.InsertCellPoint(1)
lines.InsertCellPoint(2)

td = vtk.vtkPolyData()
td.SetPoints(pts)
td.SetLines(lines)

tubef = vtk.vtkTubeFilter()
tubef.SetInput(td)
tubef.SetRadius(0.1)
tubef.SetNumberOfSides(50)
tubef.Update()

tubemapper = vtk.vtkPolyDataMapper()
tubemapper.SetInput(tubef.GetOutput())
tubeActor = vtk.vtkActor()
tubeActor.SetMapper(tubemapper)

# lighting
light1 = vtk.vtkLight()
light1.SetIntensity(.7)
#light.SetLightTypeToSceneLight()
light1.SetPosition(1, -1, 1)
#light.SetFocalPoint(0, 0, 0)
#light.SetPositional(True)
#light.SetConeAngle(60)
light1.SetDiffuseColor(1, 1, 1)
#light.SetAmbientColor(0, 1, 0)

light2 = vtk.vtkLight()
light2.SetIntensity(.7)
#light.SetLightTypeToSceneLight()
light2.SetPosition(-1, -1, 1)
#light.SetFocalPoint(0, 0, 0)
#light.SetPositional(True)
#light.SetConeAngle(60)
light2.SetDiffuseColor(1, 1, 1)
#light.SetAmbientColor(0, 1, 0)

light3 = vtk.vtkLight()
light3.SetIntensity(.7)
#light.SetLightTypeToSceneLight()
light3.SetPosition(-1, -1, -1)
#light.SetFocalPoint(0, 0, 0)
#light.SetPositional(True)
#light.SetConeAngle(60)
light3.SetDiffuseColor(1, 1, 1)
#light.SetAmbientColor(0, 1, 0)


# Visualize
renderer = vtk.vtkRenderer()
# set viewpoint randomly
renderer.SetActiveCamera(camera)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(600, 600)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# renderer.AddActor(part1Actor)
renderer.AddActor(actorx)
renderer.AddActor(actory)
renderer.AddActor(actorz)
renderer.AddActor(tubeActor)

#renderer.RemoveAllViewProps()

renderer.SetBackground(0.1, 0.1, 0.1) # Background color
renderer.SetAmbient(.4, .4, .4)
#renderer.TwoSidedLightingOff()
#renderer.LightFollowCameraOff()
renderer.AddLight(light1)
renderer.AddLight(light2)
renderer.AddLight(light3)
renderWindow.Render()

"""
vrml_exporter = vtk.vtkVRMLExporter()
vrml_exporter.SetInput(renderWindow)
vrml_exporter.SetFileName('test.wrl')
vrml_exporter.Write()

obj_exporter = vtk.vtkOBJExporter()
obj_exporter.SetInput(renderWindow)
obj_exporter.SetFilePrefix('test')
obj_exporter.Write()
"""

renderWindowInteractor.Start()

