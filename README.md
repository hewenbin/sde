[Surface density estimate (SDE)](https://ieeexplore.ieee.org/document/8525340) [1] extends [kernel density estimate (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation) [2] from discrete data points to surfaces (i.e., polygon meshes) to model the positional uncertainty of input surfaces.

The aim of this project is to provide OpenGL and WebGL based parallel implementations on computing SDE for a set of input surfaces.

[A WebGL based live demo](https://hewenbin.github.io/sde/javascript/examples/sde.html)

### Dependencies

The core functionalities (e.g., SDE computation) of this project are self-contained. For the visualization of the density estimation results in the examples, third-party visualization libraries (e.g., [vtk](https://www.vtk.org/) and [sharevol.js](https://github.com/OKaluza/sharevol)) are used. Note that sharevol.js used in the examples is slightly different from the original version to address new requirements of this project.

### Usage

##### WebGL and JavaScript

Download [sde.js](https://raw.githubusercontent.com/hewenbin/sde/master/javascript/sde.js) and include it in your HTML.

```html
<script src="lib/sde.js"></script>
```

The following code computes the SDE of a triangle with respect to the input parameters (e.g., bandwidth matrix).

```javascript
// input surfaces
var surfaces = [-8., -3., 2.,    // first vertex
                7.,  -9., 4.,    // second vertex
                1.,  8.,  -2.];  // ...

// parameters
var xmin = -10., ymin = -10., zmin = -10.,  // Physical domain that
    xmax =  10., ymax =  10., zmax =  10.;  // density estimation is
                                            // performed on.

var xdim = 64, ydim = 64, zdim = 64;  // Grid resolution of
                                      // the physical domain.

var H = [[.01, 0.,  0.],
         [0.,  .01, 0.],
         [0.,  0.,  .01]];  // bandwidth matrix

// surface density estimator
var estimator = new DensityEstimator(surfaces);

// SDE computation
var sde = estimator.Compute(xmin, ymin, zmin,
                            xmax, ymax, zmax,
                            xdim, ydim, zdim,
                            H);
```

If everything went well, you should get an array of densities for the positions of interest.

##### OpenGL and C/C++

Under development.

### References

[1] Wenbin He, Hanqi Guo, Han-Wei Shen, and Tom Peterka, "eFESTA: Ensemble Feature Exploration with Surface Density Estimates", IEEE Transactions on Visualization and Computer Graphics.

[2] Kernel density estimation. https://en.wikipedia.org/wiki/Kernel_density_estimation.
