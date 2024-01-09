size_t igroup() {
  const size_t ngx = get_num_groups(0);
  const size_t gx = get_group_id(0);
  const size_t gy = get_group_id(1);
  return gy * ngx + gx;
}

size_t iglobal(int lead_y) {
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  return iy * lead_y + ix;
}

size_t iglobal_ixy(int lead_y, int ix, int iy) {
  return iy * lead_y + ix;
}

////////////////////////////////////////
// Assignment operations.
////////////////////////////////////////

__kernel void assign_fill(int lead_y, __global Scal* u, Scal value) {
  const size_t i = iglobal(lead_y);
  u[i] = value;
}

__kernel void assign_add(int lead_y, __global Scal* u, __global const Scal* v) {
  const size_t i = iglobal(lead_y);
  u[i] += v[i];
}

__kernel void assign_sub(int lead_y, __global Scal* u, __global const Scal* v) {
  const size_t i = iglobal(lead_y);
  u[i] -= v[i];
}

__kernel void assign_subarray(int lead_y, __global Scal* u,
                              __global const Scal* v, int ix_u, int iy_u,
                              int ix_v, int iy_v, int ix_cnt, int iy_cnt) {
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  if (ix < ix_cnt && iy < iy_cnt) {
    u[(iy_u + iy) * lead_y + (ix_u + ix)] +=
        v[(iy_v + iy) * lead_y + (ix_v + ix)];
  }
}

////////////////////////////////////////
// Reduction operations.
////////////////////////////////////////

__kernel void reduce_max(int lead_y, __global const Scal* u,
                         __global Scal* output) {
  const size_t i = iglobal(lead_y);
  output[igroup()] = work_group_reduce_max(u[i]);
}

__kernel void reduce_min(int lead_y, __global const Scal* u,
                         __global Scal* output) {
  const size_t i = iglobal(lead_y);
  output[igroup()] = work_group_reduce_min(u[i]);
}

__kernel void reduce_sum(int lead_y, __global const Scal* u,
                         __global Scal* output) {
  const size_t i = iglobal(lead_y);
  output[igroup()] = work_group_reduce_add(u[i]);
}

__kernel void reduce_dot(int lead_y, __global const Scal* u,
                         __global const Scal* v, __global Scal* output) {
  const size_t i = iglobal(lead_y);
  output[igroup()] = work_group_reduce_add(u[i] * v[i]);
}

////////////////////////////////////////
// Unary operations.
////////////////////////////////////////

__kernel void scalar_add(int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] + v;
}

__kernel void scalar_sub(int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] - v;
}

__kernel void scalar_sub2(int lead_y, Scal u, __global const Scal* v,
                          __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u - v[i];
}

__kernel void scalar_mul(int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] * v;
}

__kernel void scalar_div(int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] / v;
}

__kernel void scalar_div2(int lead_y, Scal u, __global const Scal* v,
                          __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u / v[i];
}

__kernel void unary_sin(int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = sin(u[i]);
}

__kernel void unary_cos(int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = cos(u[i]);
}

__kernel void unary_exp(int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = exp(u[i]);
}

__kernel void unary_log(int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = log(u[i]);
}

__kernel void unary_sqr(int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] * u[i];
}

__kernel void unary_sqrt(int lead_y, __global const Scal* u,
                         __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = sqrt(u[i]);
}

// shift_x and shift_y must be within [0, nx - 1], [0, ny - 1].
__kernel void unary_roll(int lead_y, __global const Scal* u, int shift_x,
                         int shift_y, __global Scal* res) {
  const size_t nx = get_global_size(0);
  const size_t ny = get_global_size(1);
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  const size_t ixs = (ix < shift_x ? ix + nx - shift_x : ix - shift_x);
  const size_t iys = (iy < shift_y ? iy + ny - shift_y : iy - shift_y);
  const size_t i = iglobal_ixy(lead_y, ix, iy);
  const size_t is = iglobal_ixy(lead_y, ixs, iys);
  res[i] = u[is];
}

// Restricts field `u` of size (nx,ny) to the next coarser level.
// res: output buffer of size (nx/2, ny/2).
__kernel void field_restrict(__global const Scal* u, int nx, int ny,
                             __global Scal* res) {
  Scal (^U)(size_t, size_t) = ^(size_t ix, size_t iy) {
    return u[iy * nx + ix];
  };
  const size_t nxc = nx / 2;
  const size_t nyc = ny / 2;
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  if (ix < nxc && iy < nyc) {
    Scal a = 0;
    for (int dx = 0; dx < 2; ++dx) {
      for (int dy = 0; dy < 2; ++dy) {
        a += U(2 * ix + dx, 2 * iy + dy);
      }
    }
    res[iy * nxc + ix] = a * 0.25;
  }
}

// Adjoint of field_restrict().
// res: output buffer of size (nx*2, ny*2).
__kernel void field_restrict_adjoint(__global const Scal* u, int nx, int ny,
                                     __global Scal* res) {
  const size_t nxf = nx * 2;
  const size_t nyf = ny * 2;
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  for (int dx = 0; dx < 2; ++dx) {
    for (int dy = 0; dy < 2; ++dy) {
      const size_t ixf = 2 * ix + dx;
      const size_t iyf = 2 * iy + dy;
      res[iyf * nxf + ixf] = u[iy * nx + ix] * 0.25;
    }
  }
}

__kernel void unary_conv(int lead_y, __global const Scal* u,  //
                         Scal a, Scal axm, Scal axp, Scal aym, Scal ayp,
                         __global Scal* res) {
  Scal (^U)(size_t, size_t) = ^(size_t ix, size_t iy) {
    return u[iglobal_ixy(lead_y, ix, iy)];
  };
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  const size_t nx = get_global_size(0);
  const size_t ny = get_global_size(1);
  const size_t ixm = (ix == 0 ? nx - 1 : ix - 1);
  const size_t ixp = (ix + 1 == nx ? 0 : ix + 1);
  const size_t iym = (iy == 0 ? ny - 1 : iy - 1);
  const size_t iyp = (iy + 1 == ny ? 0 : iy + 1);
  const size_t i = iglobal_ixy(lead_y, ix, iy);

  res[i] = a * U(ix, iy) + axm * U(ixm, iy) + axp * U(ixp, iy) +
           aym * U(ix, iym) + ayp * U(ix, iyp);
}

////////////////////////////////////////
// Binary operations.
////////////////////////////////////////

__kernel void field_add(int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] + v[i];
}

__kernel void field_sub(int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] - v[i];
}

__kernel void field_mul(int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] * v[i];
}

__kernel void field_div(int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(lead_y);
  res[i] = u[i] / v[i];
}
