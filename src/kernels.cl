size_t igroup() {
  const size_t ngx = get_num_groups(0);
  const size_t gx = get_group_id(0);
  const size_t gy = get_group_id(1);
  return gy * ngx + gx;
}

size_t iglobal(int start, int lead_y) {
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  return start + iy * lead_y + ix;
}

////////////////////////////////////////
// Assignment operations.
////////////////////////////////////////

__kernel void assign_fill(int start, int lead_y, __global Scal* u, Scal value) {
  const size_t i = iglobal(start, lead_y);
  u[i] = value;
}

__kernel void assign_add(int start, int lead_y, __global Scal* u,
                         __global const Scal* v) {
  const size_t i = iglobal(start, lead_y);
  u[i] += v[i];
}

__kernel void assign_sub(int start, int lead_y, __global Scal* u,
                         __global const Scal* v) {
  const size_t i = iglobal(start, lead_y);
  u[i] -= v[i];
}

__kernel void assign_subarray(int start, int lead_y, __global Scal* u,
                              __global const Scal* v, int ix_u, int iy_u,
                              int ix_v, int iy_v, int ix_cnt, int iy_cnt) {
  const size_t ix = get_global_id(0);
  const size_t iy = get_global_id(1);
  if (ix < ix_cnt && iy < iy_cnt) {
    u[start + (iy_u + iy) * lead_y + (ix_u + ix)] +=
        v[start + (iy_v + iy) * lead_y + (ix_v + ix)];
  }
}

////////////////////////////////////////
// Reduction operations.
////////////////////////////////////////

__kernel void reduce_max(int start, int lead_y, __global const Scal* u,
                         __global Scal* output) {
  const size_t i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_max(u[i]);
}

__kernel void reduce_min(int start, int lead_y, __global const Scal* u,
                         __global Scal* output) {
  const size_t i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_min(u[i]);
}

__kernel void reduce_sum(int start, int lead_y, __global const Scal* u,
                         __global Scal* output) {
  const size_t i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_add(u[i]);
}

__kernel void reduce_dot(int start, int lead_y, __global const Scal* u,
                         __global const Scal* v, __global Scal* output) {
  const size_t i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_add(u[i] * v[i]);
}

////////////////////////////////////////
// Unary operations.
////////////////////////////////////////

__kernel void scalar_add(int start, int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] + v;
}

__kernel void scalar_sub(int start, int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] - v;
}

__kernel void scalar_sub2(int start, int lead_y, Scal u, __global const Scal* v,
                          __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u - v[i];
}

__kernel void scalar_mul(int start, int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] * v;
}

__kernel void scalar_div(int start, int lead_y, __global const Scal* u, Scal v,
                         __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] / v;
}

__kernel void scalar_div2(int start, int lead_y, Scal u, __global const Scal* v,
                          __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u / v[i];
}

__kernel void unary_sin(int start, int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = sin(u[i]);
}

__kernel void unary_cos(int start, int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = cos(u[i]);
}

__kernel void unary_exp(int start, int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = exp(u[i]);
}

__kernel void unary_log(int start, int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = log(u[i]);
}

__kernel void unary_sqr(int start, int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] * u[i];
}

__kernel void unary_sqrt(int start, int lead_y, __global const Scal* u,
                        __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = sqrt(u[i]);
}

////////////////////////////////////////
// Binary operations.
////////////////////////////////////////

__kernel void field_add(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] + v[i];
}

__kernel void field_sub(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] - v[i];
}

__kernel void field_mul(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] * v[i];
}

__kernel void field_div(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const size_t i = iglobal(start, lead_y);
  res[i] = u[i] / v[i];
}
