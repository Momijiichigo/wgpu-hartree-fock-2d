// Plan: use (32, 32, 1) workgroups
// workgroup_size(64, 4, 1)
// mapped to 512x512 2D array
// * For thread size dependent parts,
// look for GROUPSIZE comments
// * For hamiltonian matrix dimension dependent parts,
// look for DIM comments

struct SystemInfo {
  a1: vec2<f32>,
  a2: vec2<f32>,
  b1: vec2<f32>,
  b2: vec2<f32>,
  a0: f32,
  b0: f32,
  delta: f32,
  t: f32,
  area_unit_cell: f32,
}

alias H2x2 = array<complex, 4>; // 2x2 matrix

struct EigenInfo {
  k_value: vec2<f32>,
  eigenvalues: array<f32, 2>,
  eigenvectors: array<array<complex, 2>, 2>,
}



@group(0) @binding(0)
var<uniform> system_info: SystemInfo;

@group(0) @binding(1)
var<storage, read_write> k_values: array<vec2<f32>, 262144>; // 512x512 GROUPSIZE

@group(0) @binding(2)
var<storage, read_write> h0: array<H2x2, 262144>; // 512x512 GROUPSIZE

@group(0) @binding(3)
var<storage, read_write> eigen_info_arr: array<EigenInfo, 262144>; // 512x512 GROUPSIZE

@group(0) @binding(4)
var<uniform> chem_potential_trial: f32;

@group(0) @binding(5)
var<storage, read_write> charge_density: array<f32, 262144>;


const Nk: u32 = 512u;

/// Complex utilities
alias complex = vec2<f32>;
const cmplx_1: complex = complex(1.0, 0.0);
const im: complex = complex(0.0, 1.0);

fn to_c(a: f32) -> complex {
  return complex(a, 0.0);
}
// complex product
fn cprod(a: complex, b: complex) -> complex {
  return complex(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}
// complex conjugate
fn conj(a: complex) -> complex {
  return complex(a.x, -a.y);
}

// inverse of complex
fn c_inv(a: complex) -> complex {
  let denom = a.x * a.x + a.y * a.y;
  return complex(
    a.x / denom,
    -a.y / denom
  );
}

fn csqrt(a: complex) -> complex {
  let r = length(a);
  let theta = atan2(a.y, a.x);
  return complex(
    sqrt(r) * cos(theta / 2.0),
    sqrt(r) * sin(theta / 2.0)
  );
}

// complex exponential
fn cexp(a: complex) -> complex {
  let r = length(a);
  let theta = atan2(a.y, a.x);
  return complex(
    r * cos(theta),
    r * sin(theta)
  );
}

// complex arr2 operations

// complex array norm
fn cnorm_arr2(v: array<complex, 2>) -> f32 {
  return sqrt(v[0].x * v[0].x + v[0].y * v[0].y + v[1].x * v[1].x + v[1].y * v[1].y);
}

// complex array dot product with real scalar
fn dot_arr2_f32(a: array<complex, 2>, b: f32) -> array<complex, 2> {
  return array<complex, 2>(
    a[0] * b,
    a[1] * b
  );
}


// Hamiltonian
// DIM
fn h(k: vec2<f32>) -> H2x2 {
  let f = cmplx_1
    + cexp(cprod(im, to_c(dot(k, system_info.a1))))
    + cexp(cprod(im, to_c(dot(k, system_info.a2))));

  let f_conj = conj(f);
  let t = to_c(system_info.t);
  return H2x2(
    to_c(system_info.delta),
    cprod(t, f),
    cprod(t, f_conj),
    to_c(-system_info.delta)
  );
}


fn hermitian_eigen(matrix2x2: H2x2) -> EigenInfo {
  // Eigenvalues
  let a = matrix2x2[0]; // complex
  let b = matrix2x2[1];
  let c = matrix2x2[2];
  let d = matrix2x2[3];

  // make sure to use complex operations e.g. cprod
  let trace = a + d;
  let det = cprod(a, d) - cprod(b, c);
  let discriminant = cprod(trace, trace) - 4.0 * det;
  let sqrt_discriminant = csqrt(discriminant);

  let eigenvalue1 = (trace + sqrt_discriminant) / 2.0;
  let eigenvalue2 = (trace - sqrt_discriminant) / 2.0;

  // Eigenvectors
  // equation to get eigenvector
  // Ax = λx
  // (A - λI)x = 0
  // (A_00 - λ)x_0 + A_01x_1 = 0
  // x_1 = -(A_00 - λ)x_0 / A_01
  let eigenvector1 = array<complex, 2>(cmplx_1, cprod(eigenvalue1 - a, c_inv(b)));
  let eigenvector2 = array<complex, 2>(cmplx_1, cprod(eigenvalue2 - a, c_inv(b)));
  // normalize Eigenvectors
  let norm1_inv = 1.0 / cnorm_arr2(eigenvector1);
  let norm2_inv = 1.0 / cnorm_arr2(eigenvector2);
  let eigenvec1 = dot_arr2_f32(eigenvector1, norm1_inv);
  let eigenvec2 = dot_arr2_f32(eigenvector2, norm2_inv);
  
  // Eigenvalues and Eigenvectors
  return EigenInfo(
    vec2<f32>(0.0, 0.0), // placeholder
    array<f32, 2>(eigenvalue1.x, eigenvalue2.x), // Use .x to get real part
    array<array<complex, 2>, 2>(
      eigenvec1,
      eigenvec2
    )
  );
}


// Fermi-Dirac distribution
// simple theta function
fn fermi_dirac(energy: f32, chem_potential: f32) -> f32 {
  if (energy < chem_potential) {
    return 1.0;
  } else {
    return 0.0;
  }
}



/// Convert 3D index to 2D index
// GROUPSIZE
fn get_xy(index: u32) -> vec2<u32> {
    // 64^2 = 4096
    let x = index % 512u;
    let y = index / 512u;
    return vec2<u32>(x, y);
}

/// Convert 3D index to 1D index
fn get_index(workgroup_pos: vec3<u32>, local_index: u32) -> u32 {
    // workgroup size: 256
    return local_index + workgroup_pos.x * 256u + workgroup_pos.y * 256u * 32u;
}

// GROUPSIZE
@compute @workgroup_size(64, 4, 1)
fn compute_k_grid(
    @builtin(workgroup_id)
    workgroup_pos: vec3<u32>,
    @builtin(local_invocation_index)
    local_index: u32,
) {
    let index = get_index(workgroup_pos, local_index);
    let grid = get_xy(index);
    // (x * b1)/num_grid1 + (y * b2) / num_grid2
    let x = f32(grid.x);
    let y = f32(grid.y);
    let Nk = f32(Nk);
    k_values[index] = x * system_info.b1 / Nk + y * system_info.b2 / Nk;
}
@compute @workgroup_size(64, 4, 1)
fn calc_initial_eigen(
    @builtin(workgroup_id)
    workgroup_pos: vec3<u32>,
    @builtin(local_invocation_index)
    local_index: u32,
) {
    let index = get_index(workgroup_pos, local_index);
    let k = k_values[index];
    // 2x2 matrix
    let hamiltonian = h(k);

    h0[index] = hamiltonian;

    // Eigenvalues and eigenvectors
    var eigen_info = hermitian_eigen(hamiltonian);
    eigen_info.k_value = k;
    // Store eigenvalues and Eigenvectors
    eigen_info_arr[index] = eigen_info;
}

@compute @workgroup_size(64, 4, 1)
fn calc_charge_density(
    @builtin(workgroup_id)
    workgroup_pos: vec3<u32>,
    @builtin(local_invocation_index)
    local_index: u32,
) {
    let index = get_index(workgroup_pos, local_index);
    let energies = eigen_info_arr[index].eigenvalues;

    let energy0 = energies[0];
    let energy1 = energies[1];
    let fermi0 = fermi_dirac(energy0, chem_potential_trial);
    let fermi1 = fermi_dirac(energy1, chem_potential_trial);

    charge_density[index] = 2.0 * (fermi0 + fermi1);
}



