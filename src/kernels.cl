int igroup() {
  const int ngx = get_num_groups(0);
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  return gy * ngx + gx;
}

int iglobal(int start, int lead_y) {
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);
  return start + iy * lead_y + ix;
}

__kernel void field_max(int start, int lead_y, __global const Scal* u,
                        __global Scal* output) {
  const int i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_max(u[i]);
}

__kernel void field_sum(int start, int lead_y, __global const Scal* u,
                        __global Scal* output) {
  const int i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_add(u[i]);
}

__kernel void field_add(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const int i = iglobal(start, lead_y);
  res[i] = u[i] + v[i];
}

__kernel void field_sub(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const int i = iglobal(start, lead_y);
  res[i] = u[i] - v[i];
}

__kernel void field_mul(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const int i = iglobal(start, lead_y);
  res[i] = u[i] * v[i];
}

__kernel void field_div(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* res) {
  const int i = iglobal(start, lead_y);
  res[i] = u[i] / v[i];
}

__kernel void field_dot(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* output) {
  const int i = iglobal(start, lead_y);
  output[igroup()] = work_group_reduce_add(u[i] * v[i]);
}

__kernel void field_fill(int start, int lead_y, __global Scal* u, Scal value) {
  const int i = iglobal(start, lead_y);
  u[i] = value;
}
