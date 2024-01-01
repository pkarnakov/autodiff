__kernel void field_max(int start, int lead_y, __global const Scal* u,
                        __global Scal* output) {
  const int ngx = get_num_groups(0);
  const int ngy = get_num_groups(1);
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);
  const int i = start + iy * lead_y + ix;
  output[gy * ngx + gx] = work_group_reduce_max(u[i]);
}

__kernel void field_sum(int start, int lead_y, __global const Scal* u,
                        __global Scal* output) {
  const int ngx = get_num_groups(0);
  const int ngy = get_num_groups(1);
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);
  const int i = start + iy * lead_y + ix;
  output[gy * ngx + gx] = work_group_reduce_add(u[i]);
}

__kernel void field_dot(int start, int lead_y, __global const Scal* u,
                        __global const Scal* v, __global Scal* output) {
  const int ngx = get_num_groups(0);
  const int ngy = get_num_groups(1);
  const int gx = get_group_id(0);
  const int gy = get_group_id(1);
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);
  const int i = start + iy * lead_y + ix;
  output[gy * ngx + gx] = work_group_reduce_add(u[i] * v[i]);
}
