size_t igroup() {
  const size_t ngx = get_num_groups(0);
  const size_t gx = get_group_id(0);
  const size_t gy = get_group_id(1);
  return gy * ngx + gx;
}

size_t iflat() {
  const size_t nx = get_global_size(0);
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  return iy * nx + ix;
}

size_t iflat_ixy(int nx, int ix, int iy) {
  return iy * nx + ix;
}

////////////////////////////////////////
// Assignment operations.
////////////////////////////////////////

__kernel void assign_fill(__global Scal* u, Scal value) {
  const size_t i = iflat();
  u[i] = value;
}

__kernel void assign_add(__global Scal* u, __global const Scal* v) {
  const size_t i = iflat();
  u[i] += v[i];
}

__kernel void assign_sub(__global Scal* u, __global const Scal* v) {
  const size_t i = iflat();
  u[i] -= v[i];
}

__kernel void assign_subarray(__global Scal* u, __global const Scal* v,
                              int ix_u, int iy_u, int ix_v, int iy_v,
                              int ix_cnt, int iy_cnt) {
  const size_t nx = get_global_size(0);
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  if (ix < ix_cnt && iy < iy_cnt) {
    u[(iy_u + iy) * nx + (ix_u + ix)] += v[(iy_v + iy) * nx + (ix_v + ix)];
  }
}

////////////////////////////////////////
// Reduction operations.
////////////////////////////////////////

__kernel void reduce_max(__global const Scal* u, __global Scal* output) {
  const size_t i = iflat();
  output[igroup()] = work_group_reduce_max(u[i]);
}

__kernel void reduce_min(__global const Scal* u, __global Scal* output) {
  const size_t i = iflat();
  output[igroup()] = work_group_reduce_min(u[i]);
}

__kernel void reduce_sum(__global const Scal* u, __global Scal* output) {
  const size_t i = iflat();
  output[igroup()] = work_group_reduce_add(u[i]);
}

__kernel void reduce_dot(__global const Scal* u, __global const Scal* v,
                         __global Scal* output) {
  const size_t i = iflat();
  output[igroup()] = work_group_reduce_add(u[i] * v[i]);
}

////////////////////////////////////////
// Unary operations.
////////////////////////////////////////

__kernel void scalar_add(__global const Scal* u, Scal v, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] + v;
}

__kernel void scalar_sub(__global const Scal* u, Scal v, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] - v;
}

__kernel void scalar_sub2(Scal u, __global const Scal* v, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u - v[i];
}

__kernel void scalar_mul(__global const Scal* u, Scal v, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] * v;
}

__kernel void scalar_div(__global const Scal* u, Scal v, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] / v;
}

__kernel void scalar_div2(Scal u, __global const Scal* v, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u / v[i];
}

__kernel void unary_sin(__global const Scal* u, __global Scal* res) {
  const size_t i = iflat();
  res[i] = sin(u[i]);
}

__kernel void unary_cos(__global const Scal* u, __global Scal* res) {
  const size_t i = iflat();
  res[i] = cos(u[i]);
}

__kernel void unary_exp(__global const Scal* u, __global Scal* res) {
  const size_t i = iflat();
  res[i] = exp(u[i]);
}

__kernel void unary_log(__global const Scal* u, __global Scal* res) {
  const size_t i = iflat();
  res[i] = log(u[i]);
}

__kernel void unary_sqr(__global const Scal* u, __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] * u[i];
}

__kernel void unary_sqrt(__global const Scal* u, __global Scal* res) {
  const size_t i = iflat();
  res[i] = sqrt(u[i]);
}

// shift_x and shift_y must be within [0, nx - 1], [0, ny - 1].
__kernel void unary_roll(__global const Scal* u, int shift_x, int shift_y,
                         __global Scal* res) {
  const size_t nx = get_global_size(0);
  const size_t ny = get_global_size(1);
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  const size_t ixs = (ix < shift_x ? ix + nx - shift_x : ix - shift_x);
  const size_t iys = (iy < shift_y ? iy + ny - shift_y : iy - shift_y);
  const size_t i = iflat_ixy(nx, ix, iy);
  const size_t is = iflat_ixy(nx, ixs, iys);
  res[i] = u[is];
}

__kernel void field_restrict(__global const Scal* u, __global Scal* res) {
  const size_t nx = get_global_size(0);
  const size_t ny = get_global_size(1);
  const size_t nxf = nx * 2;
  const size_t nyf = ny * 2;
  Scal (^U)(size_t, size_t) = ^(size_t ixf, size_t iyf) {
    return u[iyf * nxf + ixf];
  };
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  Scal a = 0;
  for (int dx = 0; dx < 2; ++dx) {
    for (int dy = 0; dy < 2; ++dy) {
      a += U(ix * 2 + dx, iy * 2 + dy);
    }
  }
  res[iy * nx + ix] = a * 0.25;
}

__kernel void field_restrict_adjoint(__global const Scal* u,
                                     __global Scal* res) {
  const size_t nx = get_global_size(0);
  const size_t ny = get_global_size(1);
  const size_t nxf = nx * 2;
  const size_t nyf = ny * 2;
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  for (int dx = 0; dx < 2; ++dx) {
    for (int dy = 0; dy < 2; ++dy) {
      const size_t ixf = ix * 2 + dx;
      const size_t iyf = iy * 2 + dy;
      res[iyf * nxf + ixf] = u[iy * nx + ix] * 0.25;
    }
  }
}

__kernel void field_interpolate(__global const Scal* u, __global Scal* res) {
  const size_t nxf = get_global_size(0);
  const size_t nyf = get_global_size(1);
  const size_t nx = nxf / 2;
  const size_t ny = nyf / 2;
  Scal (^interp)(size_t, size_t, size_t, size_t, Scal, Scal) =
      ^(size_t ix, size_t ixp, size_t iy, size_t iyp, Scal wx, Scal wy) {
        const Scal u00 = u[iy * nx + ix];
        const Scal u10 = u[iy * nx + ixp];
        const Scal u01 = u[iyp * nx + ix];
        const Scal u11 = u[iyp * nx + ixp];
        return (u00 * (1 - wx) + u10 * wx) * (1 - wy) +
               (u01 * (1 - wx) + u11 * wx) * wy;
      };
  const size_t ixf = get_global_id(0);
  const size_t iyf = get_global_id(1);
  if (ixf > 0 && iyf > 0 && ixf + 1 < nxf && iyf + 1 < nyf) {
    // Inner cells.
    const size_t ix = (ixf - 1) / 2;
    const size_t iy = (iyf - 1) / 2;
    const size_t dx = (ixf - 1) % 2;
    const size_t dy = (iyf - 1) % 2;
    const Scal wx = 0.25 + dx * 0.5;
    const Scal wy = 0.25 + dy * 0.5;
    res[iyf * nxf + ixf] = interp(ix, ix + 1, iy, iy + 1, wx, wy);
  } else {
    // Boundary cells.
    const size_t ix = (ixf == 0 || nx == 1 ? 0
                       : ixf == nxf - 1    ? nx - 2
                                           : (ixf - 1) / 2);
    const size_t iy = (iyf == 0 || ny == 1 ? 0
                       : iyf == nyf - 1    ? ny - 2
                                           : (iyf - 1) / 2);
    const size_t ixp = (ix + 1 < nx ? ix + 1 : ix);
    const size_t iyp = (iy + 1 < ny ? iy + 1 : iy);
    const Scal wx = 0.5 * (ixf - ix * 2) - 0.25;
    const Scal wy = 0.5 * (iyf - iy * 2) - 0.25;
    res[iyf * nxf + ixf] = interp(ix, ixp, iy, iyp, wx, wy);
  }
}

__kernel void field_interpolate_adjoint(__global const Scal* u,
                                        __global Scal* res) {
  const int nx = get_global_size(0);
  const int ny = get_global_size(1);
  const int nxf = nx * 2;
  const int nyf = ny * 2;
  Scal (^U)(int, int) = ^(int ixf, int iyf) {
    return u[iyf * nxf + ixf];
  };
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);
  Scal usum = 0;
  // Iterates over dx=[-1,2] in inner cells, and less near the boundaries.
  for (int dy = -min(iy, 1); dy < 2 + min(ny - 1 - iy, 1); ++dy) {
    for (int dx = -min(ix, 1); dx < 2 + min(nx - 1 - ix, 1); ++dx) {
      const int ixf = ix * 2 + dx;
      const int iyf = iy * 2 + dy;
      const Scal wx = 1 - 0.5 * fabs(dx - 0.5);
      const Scal wy = 1 - 0.5 * fabs(dy - 0.5);
      usum += wx * wy * U(ixf, iyf);
    }
  }
  // Edges.
  for (int dx = -min(ix, 1); dx < 2 + min(nx - 1 - ix, 1); ++dx) {
    const Scal wx = 1 - 0.5 * fabs(dx - 0.5);
    const int ixf = ix * 2 + dx;
    if (iy <= 1) {
      usum += (0.5 - iy * 0.75) * U(ixf, 0) * wx;
    }
    if (ny - 1 - iy <= 1) {
      usum += (0.5 - (ny - 1 - iy) * 0.75) * U(ixf, nyf - 1) * wx;
    }
  }
  for (int dy = -min(iy, 1); dy < 2 + min(ny - 1 - iy, 1); ++dy) {
    const Scal wy = 1 - 0.5 * fabs(dy - 0.5);
    const int iyf = iy * 2 + dy;
    if (ix <= 1) {
      usum += (0.5 - ix * 0.75) * U(0, iyf) * wy;
    }
    if (nx - 1 - ix <= 1) {
      usum += (0.5 - (nx - 1 - ix) * 0.75) * U(nxf - 1, iyf) * wy;
    }
  }
#define corner(mx, my, value)            \
  do {                                   \
    if (mx <= 1 && my <= 1) {            \
      const Scal w[] = {4, -2, 1};       \
      usum += value * (w[mx + my] / 16); \
    }                                    \
  } while (0);
  corner(ix, iy, U(0, 0));
  corner(nx - 1 - ix, iy, U(nxf - 1, 0));
  corner(ix, ny - 1 - iy, U(0, nyf - 1));
  corner(nx - 1 - ix, ny - 1 - iy, U(nxf - 1, nyf - 1));
  res[iy * nx + ix] = usum;
}

__kernel void unary_conv(__global const Scal* u,  //
                         Scal a, Scal axm, Scal axp, Scal aym, Scal ayp,
                         __global Scal* res) {
  const size_t nx = get_global_size(0);
  const size_t ny = get_global_size(1);
  Scal (^U)(size_t, size_t) = ^(size_t ix, size_t iy) {
    return u[iflat_ixy(nx, ix, iy)];
  };
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  const size_t ixm = (ix == 0 ? nx - 1 : ix - 1);
  const size_t ixp = (ix + 1 == nx ? 0 : ix + 1);
  const size_t iym = (iy == 0 ? ny - 1 : iy - 1);
  const size_t iyp = (iy + 1 == ny ? 0 : iy + 1);
  const size_t i = iflat_ixy(nx, ix, iy);

  res[i] = a * U(ix, iy) + axm * U(ixm, iy) + axp * U(ixp, iy) +
           aym * U(ix, iym) + ayp * U(ix, iyp);
}

////////////////////////////////////////
// Binary operations.
////////////////////////////////////////

__kernel void field_add(__global const Scal* u, __global const Scal* v,
                        __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] + v[i];
}

__kernel void field_sub(__global const Scal* u, __global const Scal* v,
                        __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] - v[i];
}

__kernel void field_mul(__global const Scal* u, __global const Scal* v,
                        __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] * v[i];
}

__kernel void field_div(__global const Scal* u, __global const Scal* v,
                        __global Scal* res) {
  const size_t i = iflat();
  res[i] = u[i] / v[i];
}
