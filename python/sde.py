# Copyright 2018 Wenbin He. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# This code includes a serial python implementation of our SDE computation
# method. The use of this python implementation is to help users understand
# the steps of the SDE computation method rather than to provide high
# performance computation of SDE. To compute SDE for large number of surfaces
# on high resolution grids, please use our parallel implementation based on
# opengl/webgl.

from __future__ import print_function

import math
import numpy as np
from scipy import random, linalg

from tqdm import tqdm

# constants and utility functions ----------------------------------------------
kInf = 1e6
kSqrt2 = math.sqrt(2.)
kSqrt2Recip = 1. / kSqrt2
kSqrt2PiRecip = 1. / math.sqrt(2. * math.pi)

def Cross(a, b):
  return np.array([a[1] * b[2] - a[2] * b[1],
                   a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]], dtype="float32")

def Length(a):
  return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

def TriangleArea(a, b, c):
  ab, ac = b - a, c - a
  return .5 * Length(Cross(ab, ac))

def PlaneOfTriangle(a, b, c):
  ab, ac = b - a, c - a

  n = Cross(ab, ac)
  n = n / Length(n)

  d = np.dot(n, a)
  return n, d

def ProjOnPlane(v, n):
  s = np.dot(v, n) / np.dot(n, n)
  nv = n * s
  return v - nv

def DistanceToPlane(v, n, d):
  return np.dot(v, n) + d

def Sign(ax, ay, bx, by, cx, cy):
  return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

def PointInTriangle(x, y, ax, ay, bx, by, cx, cy):
  b1 = Sign(x, y, ax, ay, bx, by) < 0.
  b2 = Sign(x, y, bx, by, cx, cy) < 0.
  b3 = Sign(x, y, cx, cy, ax, ay) < 0.
  return ((b1 == b2) and (b2 == b3))

def NormPdf(x):
  return kSqrt2PiRecip * math.exp(-.5 * x * x)

def NormCdf(x):
  return .5 * (1. + math.erf(x * kSqrt2Recip))

def RandTri(a, b, c):
  r1, r2 = random.rand(), random.rand()
  return (1. - math.sqrt(r1)) * a + \
         (math.sqrt(r1) * (1. - r2)) * b + \
         r2 * math.sqrt(r1) * c

# Gammai = 15
# muli = [1]
# for i in range(1, Gammai):
#   muli.append(muli[i - 1] * i)

# def Gamma(h, a):
#   if (h > 4.76):
#     return 0.

#   if (a < 0.):
#     return -Gamma(h, -a)
#   if (a > 1.):
#     return .5 * NormCdf(h) + .5 * NormCdf(a * h) - \
#            NormCdf(h) * NormCdf(a * h) - Gamma(a * h, 1. / a)

#   res = math.atan(a) * .5 / math.pi
#   for j in range(Gammai):
#     tmp = 0.
#     for i in range(j + 1):
#       tmp += pow(h, 2 * i) / pow(2., i) / muli[i]

#     res -= .5 / math.pi * pow(a, 2 * j + 1) * pow(-1., j) / (2. * j + 1.) * \
#            (1. - math.exp(-.5 * h * h) * tmp)
#   return res

gamma_table = np.fromfile("../gamma_table.raw", dtype="float32")
gamma_table = gamma_table.reshape((477, 101))

def Gamma(h, a):
  if (h > 4.76):
    return 0.

  if (a < 0.):
    return -Gamma(h, -a)
  if (a > 1.):
    return .5 * NormCdf(h) + .5 * NormCdf(a * h) - \
           NormCdf(h) * NormCdf(a * h) - Gamma(a * h, 1. / a)

  x, y = h * 100., a * 100.
  i, j = int(x), int(y)

  if i >= 476:
    i = 475
  if j >= 100:
    j = 99

  tx, ty = x - i, y - j
  g0 = tx * gamma_table[i + 1, j] + (1. - tx) * gamma_table[i, j]
  g1 = tx * gamma_table[i + 1, j + 1] + (1. - tx) * gamma_table[i, j + 1]

  return ty * g1 + (1. - ty) * g0
# ------------------------------------------------------------------------------

# input surfaces ---------------------------------------------------------------
verts = np.array([-0.8, -0.3,  0.2,
                   0.7, -0.9,  0.4,
                   0.1,  0.8, -0.2], dtype="float32")
# ------------------------------------------------------------------------------

# parameters -------------------------------------------------------------------
xmin, ymin, zmin = -1., -1., -1.
xmax, ymax, zmax = 1., 1., 1.  # Physical domain that density estimation is performed on.
xdim, ydim, zdim = 65, 65, 65  # Grid resolution of the physical domain.

H = np.zeros((3, 3), dtype="float32")
H[0, 0], H[0, 1], H[0, 2] = .0025,  0.,     0.
H[1, 0], H[1, 1], H[1, 2] = 0.,     .0025,  0.
H[2, 0], H[2, 1], H[2, 2] = 0.,     0.,     .0025
# ------------------------------------------------------------------------------

# SDE computation --------------------------------------------------------------
a, b, c = verts[0:3], verts[3:6], verts[6:9]  # three vertices of the triangle
area = TriangleArea(a, b, c)  # area of the triangle

Hi = linalg.inv(H)
Hi_sqrt = linalg.sqrtm(Hi)
Hi_sqrt_det = linalg.det(Hi_sqrt)

# Create a numpy array to store SDE.
sde = np.zeros([zdim, ydim, xdim], dtype="float32")

# Compute SDE for each grid point.
alpha = [0., 0., 0.]  # Used to store the bivariate normal integral over the area defined by each edge.
for i in tqdm(range(xdim * ydim * zdim)):
  # physical position of the grid point
  gk = math.floor(i / (xdim * ydim))
  gj = math.floor((i - gk * xdim * ydim) / xdim)
  gi = i % xdim

  x = np.array([gi * (xmax - xmin) / (xdim - 1.) + xmin,
                gj * (ymax - ymin) / (ydim - 1.) + ymin,
                gk * (zmax - zmin) / (zdim - 1.) + zmin], dtype="float32")

  # Transform the triangle based on the grid point position and the bandwidth matrix.
  ax, bx, cx = x - a, x - b, x - c

  a_prime = np.dot(Hi_sqrt, ax)
  b_prime = np.dot(Hi_sqrt, bx)
  c_prime = np.dot(Hi_sqrt, cx)

  area_prime = TriangleArea(a_prime, b_prime, c_prime)

  ct = area / area_prime * Hi_sqrt_det

  # Map the triangle into 2D.
  n, d = PlaneOfTriangle(a_prime, b_prime, c_prime)
  a_proj = ProjOnPlane(a_prime, n)
  b_proj = ProjOnPlane(b_prime, n)
  c_proj = ProjOnPlane(c_prime, n)

  uaxis = a_proj - b_proj
  uaxis = uaxis / Length(uaxis)
  vaxis = Cross(uaxis, n)

  udot_recip = 1. / np.dot(uaxis, uaxis)
  vdot_recip = 1. / np.dot(vaxis, vaxis)

  au = np.dot(a_proj, uaxis) * udot_recip
  av = np.dot(a_proj, vaxis) * vdot_recip

  bu = np.dot(b_proj, uaxis) * udot_recip
  bv = np.dot(b_proj, vaxis) * vdot_recip

  cu = np.dot(c_proj, uaxis) * udot_recip
  cv = np.dot(c_proj, vaxis) * vdot_recip

  lamb = DistanceToPlane(np.zeros(3, dtype="float32"), n, d)

  # Compute SDE based on the bivariate normal integral.
  if (au != 0. or av != 0.) and \
     (bu != 0. or bv != 0.) and \
     (cu != 0. or cv != 0.):
    # segment bc
    cbu, cbv = cu - bu, cv - bv
    ad = abs(bu * cv - bv * cu)
    h = ad / math.sqrt(cbu * cbu + cbv * cbv)
    if (ad > 0.):
      a1 = abs(bu * cbu + bv * cbv) / ad
      a2 = abs(cu * cbu + cv * cbv) / ad
    else:
      a1 = kInf
      a2 = kInf

    dot1 = -bu * cbu - bv * cbv
    dot2 = cu * cbu + cv * cbv
    if dot1 < 0. or dot2 < 0.:
      if (a2 > a1):
        alpha[0] = Gamma(h, a2) - Gamma(h, a1)
      else:
        alpha[0] = Gamma(h, a1) - Gamma(h, a2)
    else:
      alpha[0] = Gamma(h, a1) + Gamma(h, a2)

    # segment ca
    acu, acv = au - cu, av - cv
    ad = abs(cu * av - cv * au)
    h = ad / math.sqrt(acu * acu + acv * acv)
    if (ad > 0.):
      a1 = abs(cu * acu + cv * acv) / ad
      a2 = abs(au * acu + av * acv) / ad
    else:
      a1 = kInf
      a2 = kInf

    dot1 = -cu * acu - cv * acv
    dot2 = au * acu + av * acv
    if dot1 < 0. or dot2 < 0.:
      if (a2 > a1):
        alpha[1] = Gamma(h, a2) - Gamma(h, a1)
      else:
        alpha[1] = Gamma(h, a1) - Gamma(h, a2)
    else:
      alpha[1] = Gamma(h, a1) + Gamma(h, a2)

    # segment ab
    bau, bav = bu - au, bv - av
    ad = abs(au * bv - av * bu)
    h = ad / math.sqrt(bau * bau + bav * bav)
    if (ad > 0.):
      a1 = abs(au * bau + av * bav) / ad
      a2 = abs(bu * bau + bv * bav) / ad
    else:
      a1 = kInf
      a2 = kInf

    dot1 = -au * bau - av * bav
    dot2 = bu * bau + bv * bav
    if dot1 < 0. or dot2 < 0.:
      if (a2 > a1):
        alpha[2] = Gamma(h, a2) - Gamma(h, a1)
      else:
        alpha[2] = Gamma(h, a1) - Gamma(h, a2)
    else:
      alpha[2] = Gamma(h, a1) + Gamma(h, a2)

    # Combine alpha bc, ca, and ab.
    if (PointInTriangle(0., 0., au, av, bu, bv, cu, cv)):
      sde[gk, gj, gi] += (1. - alpha[0] - alpha[1] - alpha[2]) * \
                         NormPdf(lamb) * ct
    else:
      anglec = []
      la = math.sqrt(au * au + av * av)
      lb = math.sqrt(bu * bu + bv * bv)
      lc = math.sqrt(cu * cu + cv * cv)
      aun, avn = au / la, av / la
      bun, bvn = bu / lb, bv / lb
      cun, cvn = cu / lc, cv / lc
      anglec.append((bun * cun + bvn * cvn))
      anglec.append((cun * aun + cvn * avn))
      anglec.append((aun * bun + avn * bvn))

      minac = anglec[0]
      minaci = 0
      if anglec[1] < minac:
        minac = anglec[1]
        minaci = 1
      if anglec[2] < minac:
        minac = anglec[2]
        minaci = 2

      if (minaci == 0):
        sde[gk, gj, gi] += abs(alpha[1] + alpha[2] - alpha[0]) * \
                           NormPdf(lamb) * ct
      elif (minaci == 1):
        sde[gk, gj, gi] += abs(alpha[0] + alpha[2] - alpha[1]) * \
                           NormPdf(lamb) * ct
      elif (minaci == 2):
        sde[gk, gj, gi] += abs(alpha[0] + alpha[1] - alpha[2]) * \
                           NormPdf(lamb) * ct

  # Handle spacial cases.
  elif au == 0. and av == 0.:
    # segment bc
    cbu, cbv = cu - bu, cv - bv
    ad = abs(bu * cv - bv * cu)
    h = ad / math.sqrt(cbu * cbu + cbv * cbv)
    if (ad > 0.):
      a1 = abs(bu * cbu + bv * cbv) / ad
      a2 = abs(cu * cbu + cv * cbv) / ad
    else:
      a1 = kInf
      a2 = kInf

    dot1 = -bu * cbu - bv * cbv
    dot2 = cu * cbu + cv * cbv
    if dot1 < 0. or dot2 < 0.:
      if (a2 > a1):
        alpha[0] = Gamma(h, a2) - Gamma(h, a1)
      else:
        alpha[0] = Gamma(h, a1) - Gamma(h, a2)
    else:
      alpha[0] = Gamma(h, a1) + Gamma(h, a2)

    lb = math.sqrt(bu * bu + bv * bv)
    lc = math.sqrt(cu * cu + cv * cv)
    bun, bvn = bu / lb, bv / lb
    cun, cvn = cu / lc, cv / lc

    sde[gk, gj, gi] += abs(abs(math.acos(bun * cun + bvn * cvn)) / \
                       math.pi * .5 - alpha[0]) * NormPdf(lamb) * ct

  elif bu == 0. and bv == 0.:
    # segment ca
    acu, acv = au - cu, av - cv
    ad = abs(cu * av - cv * au)
    h = ad / math.sqrt(acu * acu + acv * acv)
    if (ad > 0.):
      a1 = abs(cu * acu + cv * acv) / ad
      a2 = abs(au * acu + av * acv) / ad
    else:
      a1 = kInf
      a2 = kInf

    dot1 = -cu * acu - cv * acv
    dot2 = au * acu + av * acv
    if dot1 < 0. or dot2 < 0.:
      if (a2 > a1):
        alpha[1] = Gamma(h, a2) - Gamma(h, a1)
      else:
        alpha[1] = Gamma(h, a1) - Gamma(h, a2)
    else:
      alpha[1] = Gamma(h, a1) + Gamma(h, a2)

    lc = math.sqrt(cu * cu + cv * cv)
    la = math.sqrt(au * au + av * av)
    cun, cvn = cu / lc, cv / lc
    aun, avn = au / la, av / la

    sde[gk, gj, gi] += abs(abs(math.acos(cun * aun + cvn * avn)) / \
                       math.pi * .5 - alpha[1]) * NormPdf(lamb) * ct

  elif cu == 0. and cv == 0.:
    # segment ab
    bau, bav = bu - au, bv - av
    ad = abs(au * bv - av * bu)
    h = ad / math.sqrt(bau * bau + bav * bav)
    if (ad > 0.):
      a1 = abs(au * bau + av * bav) / ad
      a2 = abs(bu * bau + bv * bav) / ad
    else:
      a1 = kInf
      a2 = kInf

    dot1 = -au * bau - av * bav
    dot2 = bu * bau + bv * bav
    if dot1 < 0. or dot2 < 0.:
      if (a2 > a1):
        alpha[2] = Gamma(h, a2) - Gamma(h, a1)
      else:
        alpha[2] = Gamma(h, a1) - Gamma(h, a2)
    else:
      alpha[2] = Gamma(h, a1) + Gamma(h, a2)

    la = math.sqrt(au * au + av * av)
    lb = math.sqrt(bu * bu + bv * bv)
    aun, avn = au / la, av / la
    bun, bvn = bu / lb, bv / lb

    sde[gk, gj, gi] += abs(abs(math.acos(aun * bun + avn * bvn)) / \
                       math.pi * .5 - alpha[2]) * NormPdf(lamb) * ct

# Normalize SDE by the area of the surface.
sde /= area
# ------------------------------------------------------------------------------

# SDE visualization ------------------------------------------------------------
import vtk

# Normalize SDE for visualization.
sde /= np.max(sde)

# Create the renderer, render window, and interactor.
ren = vtk.vtkRenderer()

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Copy SDE into a vtkImage.
reader = vtk.vtkImageImport()
reader.CopyImportVoidPointer(sde, sde.nbytes)
reader.SetDataScalarTypeToFloat()
reader.SetDataExtent(0, xdim - 1, 0, ydim - 1, 0, zdim - 1)
reader.SetWholeExtent(0, xdim - 1, 0, ydim - 1, 0, zdim - 1)
# reader.SetDataOrigin(xmin, ymin, zmin)
# reader.SetDataSpacing((xmax - xmin) / (xdim - 1.),
#                       (ymax - ymin) / (ydim - 1.),
#                       (zmax - zmin) / (zdim - 1.))

# The volume will be displayed by ray-cast alpha compositing.
volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetInputConnection(reader.GetOutputPort())
volumeMapper.SetBlendModeToComposite()

# Create transfer function mapping scalar value to opacity.
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0., 0.)
opacityTransferFunction.AddPoint(1., 1.)

# Create transfer function mapping scalar value to color.
colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(0.,    .188235, .164706, .470588)
colorTransferFunction.AddRGBPoint(.0625, .145098, .192157, .490196)
colorTransferFunction.AddRGBPoint(.125,  .133333, .227451, .501961)
colorTransferFunction.AddRGBPoint(.1875, .121569, .258824, .509804)
colorTransferFunction.AddRGBPoint(.25,   .117647, .321569, .521569)
colorTransferFunction.AddRGBPoint(.3125, .12549,  .396078, .529412)
colorTransferFunction.AddRGBPoint(.375,  .133333, .458824, .541176)
colorTransferFunction.AddRGBPoint(.4375, .141176, .517647, .54902 )
colorTransferFunction.AddRGBPoint(.5,    .172549, .568627, .537255)
colorTransferFunction.AddRGBPoint(.5625, .203922, .6,      .486275)
colorTransferFunction.AddRGBPoint(.625,  .231373, .639216, .435294)
colorTransferFunction.AddRGBPoint(.6875, .278431, .658824, .392157)
colorTransferFunction.AddRGBPoint(.75,   .337255, .721569, .337255)
colorTransferFunction.AddRGBPoint(.8125, .513725, .768627, .384314)
colorTransferFunction.AddRGBPoint(.875,  .690196, .819608, .435294)
colorTransferFunction.AddRGBPoint(.9375, .847059, .858824, .482353)
colorTransferFunction.AddRGBPoint(1.,    .901961, .878431, .556863)

# The VolumeProperty attaches the color and opacity functions to the
# volume, and sets other volume properties.
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.SetInterpolationTypeToLinear()

# The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
# and orientation of the volume in world coordinates.
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# Outline of the volume.
outline = vtk.vtkOutlineFilter()
outline.SetInputConnection(reader.GetOutputPort())
outlineMapper = vtk.vtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())
outlineActor = vtk.vtkActor()
outlineActor.SetMapper(outlineMapper)
outlineActor.GetProperty().SetColor(0., 0., 0.)
outlineActor.GetProperty().SetLineWidth(2.)

ren.AddActor(outlineActor)
ren.AddVolume(volume)
ren.SetBackground(.94, .94, .94)
ren.GetActiveCamera().Azimuth(45)
ren.GetActiveCamera().Elevation(30)
ren.ResetCameraClippingRange()
ren.ResetCamera()

renWin.SetSize(600, 600)
renWin.Render()
renWin.SetWindowName("SDE of a triangle")

iren.Start()
# ------------------------------------------------------------------------------
