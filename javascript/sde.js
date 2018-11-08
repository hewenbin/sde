// Copyright 2018 Wenbin He. All rights reserved.
// Use of this source code is governed by a MIT-style license that can be
// found in the LICENSE file.

function SDEstimator(verts) {
  // public variables and functions --------------------------------------------
  // set surfaces
  this.nverts = 0;
  this.area = 0.;
  this.verts_textures = [];

  // local references of canvas, gl, and program
  var canvas_ = SDEstimator.canvas,
      gl_ = SDEstimator.gl,
      program_ = SDEstimator.program;

  this.SetSurfaces = function(verts) {
    // reset surfaces
    this.nverts = 0;
    this.area = 0.;
    for (var i = 0, il = this.verts_textures.length; i < il; ++i)
      gl_.deleteTexture(this.verts_textures[i]);
    this.verts_textures.length = 0;

    // Only keep triangles whose areas are greater than zero.
    var valid_verts = [];
    if (verts !== undefined) {
      for (var i = 0, il = verts.length; i < il; i += 3) {
        var area = TriangleArea(verts[i], verts[i + 1], verts[i + 2]);
        if (area > 0.) {
          valid_verts.push(verts[i].slice());
          valid_verts.push(verts[i + 1].slice());
          valid_verts.push(verts[i + 2].slice());
          this.area += area;
        }
      }
    }

    this.nverts = valid_verts.length;
    console.log("Number of valid triangles: " + this.nverts / 3);
    console.log("Total area: " + this.area);

    // Copy the surfaces into textures.
    var max_texture_size = gl_.getParameter(gl_.MAX_TEXTURE_SIZE);
    // console.log("WebGL2 maximum texture size: " +
    //             max_texture_size + " x " + max_texture_size);
    while (valid_verts.length > 0) {
      var curr_verts = valid_verts.splice(0, Math.min(valid_verts.length,
                                                      max_texture_size * 3));

      var texture = gl_.createTexture();
      gl_.bindTexture(gl_.TEXTURE_2D, texture);
      gl_.texStorage2D(gl_.TEXTURE_2D, 1, gl_.RGB32F, 3, curr_verts.length / 3);

      gl_.texParameteri(gl_.TEXTURE_2D, gl_.TEXTURE_MIN_FILTER, gl_.NEAREST);
      gl_.texParameteri(gl_.TEXTURE_2D, gl_.TEXTURE_MAG_FILTER, gl_.NEAREST);
      gl_.texParameteri(gl_.TEXTURE_2D, gl_.TEXTURE_WRAP_S, gl_.CLAMP_TO_EDGE);
      gl_.texParameteri(gl_.TEXTURE_2D, gl_.TEXTURE_WRAP_T, gl_.CLAMP_TO_EDGE);

      gl_.texSubImage2D(gl_.TEXTURE_2D, 0, 0, 0,
                        3, curr_verts.length / 3,
                        gl_.RGB, gl_.FLOAT,
                        new Float32Array(curr_verts.flat()));

      gl_.bindTexture(gl_.TEXTURE_2D, null);

      this.verts_textures.push(texture);
    }
  }

  this.SetSurfaces(verts);

  // SDE computation
  this.Compute = function(xmin, ymin, zmin,
                          xmax, ymax, zmax,  // Physical domain that density estimation is performed on.
                          xdim, ydim, zdim,  // Grid resolution of the physical domain.
                          H) {  // bandwidth matrix
    var Hi = MatInv(H);  // inverse of the bandwidth matrix
    if (Hi === undefined) {
      alert("The input bandwidth matrix is not invertible.");
      return;
    }

    var Hi_sqrt = MatSqrt(Hi);  // square root of Hi (i.e., Hi_sqrt x Hi_sqrt = Hi)
    if (Hi_sqrt === undefined) {
      alert("The input bandwidth matrix is not positive semidefinite.");
      return;
    }
    Hi_sqrt = new Float32Array(Hi_sqrt.flat());

    // Create a 3D texture to store the density estimation result.
    var sde_texture = gl_.createTexture();
    gl_.bindTexture(gl_.TEXTURE_3D, sde_texture);
    gl_.texStorage3D(gl_.TEXTURE_3D, 1, gl_.R32F, xdim, ydim, zdim);

    gl_.texParameteri(gl_.TEXTURE_3D, gl_.TEXTURE_MIN_FILTER, gl_.NEAREST);
    gl_.texParameteri(gl_.TEXTURE_3D, gl_.TEXTURE_MAG_FILTER, gl_.NEAREST);
    gl_.texParameteri(gl_.TEXTURE_3D, gl_.TEXTURE_WRAP_S, gl_.CLAMP_TO_EDGE);
    gl_.texParameteri(gl_.TEXTURE_3D, gl_.TEXTURE_WRAP_T, gl_.CLAMP_TO_EDGE);

    gl_.bindTexture(gl_.TEXTURE_3D, null);

    // Create a framebuffer.
    var fbo = gl_.createFramebuffer();
    gl_.bindFramebuffer(gl_.FRAMEBUFFER, fbo);
    gl_.framebufferTextureLayer(
        gl_.FRAMEBUFFER, gl_.COLOR_ATTACHMENT0, sde_texture, 0, 0);

    if (gl_.checkFramebufferStatus(gl_.FRAMEBUFFER) !=
        gl_.FRAMEBUFFER_COMPLETE) {
      alert("The framebuffer is not complete.");
      return;
    }

    gl_.bindFramebuffer(gl_.FRAMEBUFFER, null);

    // setup viewport
    gl_.viewport(0, 0, xdim, ydim);

    // computation
    gl_.useProgram(program_);

    var aPosLoc = gl_.getAttribLocation(program_, "aPos");
    var aUvLoc = gl_.getAttribLocation(program_, "aUv");

    gl_.enableVertexAttribArray(aPosLoc);
    gl_.enableVertexAttribArray(aUvLoc);

    var quad = new Float32Array([-1., -1., 0., 0., 1., -1., 1., 0.,
                                 -1., 1., 0., 1., 1., 1., 1., 1.]);
    var vbo = gl_.createBuffer();
    gl_.bindBuffer(gl_.ARRAY_BUFFER, vbo);
    gl_.bufferData(gl_.ARRAY_BUFFER, quad, gl_.STATIC_DRAW);
    gl_.vertexAttribPointer(aPosLoc, 2, gl_.FLOAT, gl_.FALSE, 16, 0);
    gl_.vertexAttribPointer(aUvLoc, 2, gl_.FLOAT, gl_.FALSE, 16, 8);

    var res = new Float32Array(zdim * ydim * xdim * 4);
    gl_.bindFramebuffer(gl_.FRAMEBUFFER, fbo);
    for (var i = 0; i < zdim; ++i) {
      gl_.framebufferTextureLayer(
          gl_.FRAMEBUFFER, gl_.COLOR_ATTACHMENT0, sde_texture, 0, i);
      gl_.drawArrays(gl_.TRIANGLE_STRIP, 0, 4);

      // Copy the density estimation result from GPU to CPU.
      gl_.readBuffer(gl_.COLOR_ATTACHMENT0);
      gl_.readPixels(0, 0, xdim, ydim, gl_.RGBA, gl_.FLOAT,
                     res, i * ydim * xdim * 4);
    }

    // Release buffers and textures
    gl_.bindFramebuffer(gl_.FRAMEBUFFER, null);
    gl_.bindBuffer(gl_.ARRAY_BUFFER, null);
    gl_.useProgram(null);

    gl_.deleteBuffer(vbo);
    gl_.deleteFramebuffer(fbo);
    gl_.deleteTexture(sde_texture);

    // Normalize SDE by the area of the surface.
    for (var i = 0, il = zdim * ydim * xdim * 4; i < il; i += 4)
      res[i] /= this.area;

    return res;
  };
  // ---------------------------------------------------------------------------

  // private variables and functions -------------------------------------------
  // constants
  var kEps = 1e-10;
  var kMaxSweeps = 64;

  // subtraction of vectors
  function VecSub(va, vb) {
    return [va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]];
  }

  // cross product of vectors
  function VecCross(va, vb) {
    return [va[1] * vb[2] - va[2] * vb[1],
            va[2] * vb[0] - va[0] * vb[2],
            va[0] * vb[1] - va[1] * vb[0]];
  }

  // length of a vector
  function VecLength(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  // area of a triangle
  function TriangleArea(a, b, c) {
    var ab = VecSub(b, a), ac = VecSub(c, a);
    return .5 * VecLength(VecCross(ab, ac));
  }

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
    if (t == 0.) return;

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
      return;

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
    if (eig === undefined) return;
    if (eig.val[0] < 0. || eig.val[1] < 0. || eig.val[2] < 0.)
      return;

    var diag = [[Math.sqrt(eig.val[0]), 0., 0.],
                [0., Math.sqrt(eig.val[1]), 0.],
                [0., 0., Math.sqrt(eig.val[2])]];

    return MatMul(MatMul(eig.vec, diag), MatTrs(eig.vec));
  }
  // ---------------------------------------------------------------------------
}

// static variables and functions ----------------------------------------------
// canvas and gl
SDEstimator.canvas = document.createElement("canvas");

SDEstimator.gl = SDEstimator.canvas.getContext("webgl2");
if (!SDEstimator.gl) {
  alert("WebGL2 is not supported by this browser.");
}

SDEstimator.ext = SDEstimator.gl.getExtension("EXT_color_buffer_float");
if (!SDEstimator.ext) {
  alert("EXT_color_buffer_float is not supported by this browser.");
}

// shaders
SDEstimator.vertex_glsl = [
  "#version 300 es",
  "precision highp float;",
  "in vec2 aPos;",
  "in vec2 aUv;",
  "out vec2 vUv;",
  "void main() {",
    "gl_Position = vec4(aPos, 0., 1.);",
    "vUv = aUv;",
  "}"
].join("\n");

SDEstimator.fragment_glsl = [
  "#version 300 es",
  "precision highp float;",
  "in vec2 vUv;",
  "out vec4 fragColor;",
  "void main() {",
    "fragColor = vec4(vUv.y * 64., 0., 0., 1.);",
  "}"
].join("\n");

SDEstimator.GetShader = function(gl, type, code) {
  var shader = gl.createShader(type);

  gl.shaderSource(shader, code);
  gl.compileShader(shader);

  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === false) {
    console.error("Shader couldn\'t compile.");
  }

  if (gl.getShaderInfoLog(shader) !== "") {
    console.warn(type === gl.VERTEX_SHADER ? "Vertex:" : "Fragment:",
                 gl.getShaderInfoLog(shader));
  }

  return shader;
}

SDEstimator.program = SDEstimator.gl.createProgram();

SDEstimator.gl.attachShader(
    SDEstimator.program,
    SDEstimator.GetShader(
        SDEstimator.gl,
        SDEstimator.gl.VERTEX_SHADER,
        SDEstimator.vertex_glsl));

SDEstimator.gl.attachShader(
    SDEstimator.program,
    SDEstimator.GetShader(
        SDEstimator.gl,
        SDEstimator.gl.FRAGMENT_SHADER,
        SDEstimator.fragment_glsl));

SDEstimator.gl.linkProgram(SDEstimator.program);
// ---------------------------------------------------------------------------
