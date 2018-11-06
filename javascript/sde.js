// Copyright 2018 Wenbin He. All rights reserved.
// Use of this source code is governed by a MIT-style license that can be
// found in the LICENSE file.

function SDEstimator(verts) {
  this.verts = verts !== undefined ? verts : [];

  // SDE computation
  this.Compute = function(xmin, ymin, zmin,
                          xmax, ymax, zmax,  // Physical domain that density estimation is performed on.
                          xdim, ydim, zdim,  // Grid resolution of the physical domain.
                          H) {  // bandwidth matrix
    var Hi = MatInv(H);  // inverse of the bandwidth matrix
    if (Hi === undefined) {
      alert("The input bandwidth matrix is not invertible.");
      return undefined;
    }

    var Hi_sqrt = MatSqrt(Hi);  // square root of Hi (i.e., Hi_sqrt x Hi_sqrt = Hi)
    if (Hi_sqrt === undefined) {
      alert("The input bandwidth matrix is not positive semidefinite.");
      return undefined;
    }
    Hi_sqrt = new Float32Array(Hi_sqrt.flat());
  };

  // constants and utility functions (private) ---------------------------------
  var kEps = 1e-10;
  var kMaxSweeps = 64;

  // multiplication of matrices
  function MatMul(ma, mb) {
    return [[ma[0][0] * mb[0][0] + ma[0][1] * mb[1][0] + ma[0][2] * mb[2][0],
             ma[0][0] * mb[0][1] + ma[0][1] * mb[1][1] + ma[0][2] * mb[2][1],
             ma[0][0] * mb[0][2] + ma[0][1] * mb[1][2] + ma[0][2] * mb[2][2]],
            [ma[1][0] * mb[0][0] + ma[1][1] * mb[1][0] + ma[1][2] * mb[2][0],
             ma[1][0] * mb[0][1] + ma[1][1] * mb[1][1] + ma[1][2] * mb[2][1],
             ma[1][0] * mb[0][2] + ma[1][1] * mb[1][2] + ma[1][2] * mb[2][2]],
            [ma[2][0] * mb[0][0] + ma[2][1] * mb[1][0] + ma[2][2] * mb[2][0],
             ma[2][0] * mb[0][1] + ma[2][1] * mb[1][1] + ma[2][2] * mb[2][1],
             ma[2][0] * mb[0][2] + ma[2][1] * mb[1][2] + ma[2][2] * mb[2][2]]];
  }

  // determinant of a matrix
  function MatDet(m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }

  // transpose of a matrix
  function MatTrs(m) {
    return [[m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]]];
  }

  // inverse of a matrix
  function MatInv(m) {
    var p00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    var p10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    var p20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    var t = m[0][0] * p00 + m[0][1] * p10 + m[0][2] * p20;
    if (t == 0.) return undefined;

    var t = 1. / t;

    return [[p00 * t,
             (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * t,
             (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * t],
            [p10 * t,
             (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * t,
             (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * t],
            [p20 * t,
             (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * t,
             (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * t]];
  }

  // eigen decomposition of a matrix
  function MatEig(m) {
    if (m[0][1] != m[1][0] ||
        m[0][2] != m[2][0] ||
        m[1][2] != m[2][1])
      return undefined;

    var res = {val : [0., 0., 0.],
               vec : [[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]]};

    var m00 = m[0][0], m11 = m[1][1], m22 = m[2][2],
        m01 = m[0][1], m02 = m[0][2], m12 = m[1][2];

    for (var i = 0; i < kMaxSweeps; ++i) {
      if ((Math.abs(m01) < kEps) && (Math.abs(m02) < kEps) &&
          (Math.abs(m12) < kEps)) break;

      if (m01 != 0.) {
        var u = (m11 - m00) * .5 / m01;
        var u2 = u * u;
        var u2p1 = u2 + 1.;
        var t = (u2p1 != u2) ?
                ((u < 0.) ? -1. : 1.) * (Math.sqrt(u2p1) - Math.abs(u)) :
                .5 / u;
        var c = 1. / Math.sqrt(t * t + 1.);
        var s = c * t;
        m00 -= t * m01;
        m11 += t * m01;
        m01 = 0.;

        var tmp = c * m02 - s * m12;
        m12 = s * m02 + c * m12;
        m02 = tmp;

        for (var j = 0; j < 3; ++j) {
          var tmp = c * res.vec[j][0] - s * res.vec[j][1];
          res.vec[j][1] = s * res.vec[j][0] + c * res.vec[j][1];
          res.vec[j][0] = tmp;
        }
      }

      if (m02 != 0.) {
        var u = (m22 - m00) * .5 / m02;
        var u2 = u * u;
        var u2p1 = u2 + 1.;
        var t = (u2p1 != u2) ?
                ((u < 0.) ? -1. : 1.) * (Math.sqrt(u2p1) - Math.abs(u)) :
                .5 / u;
        var c = 1. / Math.sqrt(t * t + 1.);
        var s = c * t;

        m00 -= t * m02;
        m22 += t * m02;
        m02 = 0.;

        var tmp = c * m01 - s * m12;
        m12 = s * m01 + c * m12;
        m01 = tmp;

        for (var j = 0; j < 3; ++j) {
          var tmp = c * res.vec[j][0] - s * res.vec[j][2];
          res.vec[j][2] = s * res.vec[j][0] + c * res.vec[j][2];
          res.vec[j][0] = tmp;
        }
      }

      if (m12 != 0.) {
        var u = (m22 - m11) * .5 / m12;
        var u2 = u * u;
        var u2p1 = u2 + 1.;
        var t = (u2p1 != u2) ?
                ((u < 0.) ? -1. : 1.) * (Math.sqrt(u2p1) - Math.abs(u)) :
                .5 / u;
        var c = 1. / Math.sqrt(t * t + 1.);
        var s = c * t;

        m11 -= t * m12;
        m22 += t * m12;
        m12 = 0.;

        var tmp = c * m01 - s * m02;
        m02 = s * m01 + c * m02;
        m01 = tmp;

        for (var j = 0; j < 3; ++j) {
          var tmp = c * res.vec[j][1] - s * res.vec[j][2];
          res.vec[j][2] = s * res.vec[j][1] + c * res.vec[j][2];
          res.vec[j][1] = tmp;
        }
      }
    }

    res.val[0] = m00;
    res.val[1] = m11;
    res.val[2] = m22;

    return res;
  }

  // square root of a matrix
  function MatSqrt(m) {
    var eig = MatEig(m);
    if (eig === undefined) return undefined;
    if (eig.val[0] < 0. || eig.val[1] < 0. || eig.val[2] < 0.)
      return undefined;

    var diag = [[Math.sqrt(eig.val[0]), 0., 0.],
                [0., Math.sqrt(eig.val[1]), 0.],
                [0., 0., Math.sqrt(eig.val[2])]];

    return MatMul(MatMul(eig.vec, diag), MatTrs(eig.vec));
  }
  // ---------------------------------------------------------------------------
}
