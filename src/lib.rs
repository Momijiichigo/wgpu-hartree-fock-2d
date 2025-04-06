use lazy_static::lazy_static;
use nalgebra::Matrix2;
use wgpu::util::DeviceExt;
mod buff_utils;

use buff_utils::BufferInfo;
// const OVERFLOW: u32 = 0xffffffff;

// Plan: use 32x32x1 workgroups, with each workgroup size 64x4x1, 
// and map those threads (262144 total) into 2D grids (512x512).

use std::f32::consts::PI;
lazy_static! {

    static ref SYSTEM_INFO: SystemInfo = {
        let a0 = 3.193;
        // a1 = a₀ * [sqrt(3), 1] / 2
        // a2 = a₀ * [sqrt(3), -1] / 2
        let a1_x = a0 * (3f32).sqrt() / 2.0;
        let a1_y = a0 / 2.0;
        let a2_x = a0 * (3f32).sqrt() / 2.0;
        let a2_y = -a0 / 2.0;
        let mat_a = Matrix2::new(
            a1_x, a1_y,
            a2_x, a2_y,
        );
        // B = 2pi * transpose(inverse(A))
        let mat_b = 2.0 * PI * mat_a.try_inverse().unwrap().transpose();
        // b1 = B[:, 1]
        // b2 = B[:, 2]
        let b1_x = mat_b[(0, 0)];
        let b1_y = mat_b[(0, 1)];
        let b2_x = mat_b[(1, 0)];
        let b2_y = mat_b[(1, 1)];
        // length of b1 (equals to length of b2, due to symmetry)
        let b0 = (b1_x * b1_x + b1_y * b1_y).sqrt();

        SystemInfo {
            area_unit_cell: 0.0,
            a0,
            a1: [a1_x, a1_y],
            a2: [a2_x, a2_y],
            b0,
            b1: [b1_x, b1_y],
            b2: [b2_x, b2_y],
            delta: 0.0,
            t: -1.0,
            _pad0: 0.0,
        }
    };
}
pub async fn run() {
    println!("System config:");
    println!("{:?}", *SYSTEM_INFO);
    let context = WgpuContext::new().await;

    calc_k_grid(&context);
    calc_initial_eigen(&context);

    let chem_potential = 0.0;
    let density_arr = get_charge_density(&context, chem_potential).await;
    let density = density_arr.iter().sum::<f32>() / (density_arr.len() as f32);
    println!("# of k points: {}", density_arr.len());
    println!("Trying chemical potential: {chem_potential}");
    println!("charge density {:?}", density);

}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SystemInfo {
    a1: [f32; 2],
    a2: [f32; 2],
    b1: [f32; 2],
    b2: [f32; 2],
    a0: f32,
    b0: f32,
    delta: f32,
    t: f32,
    area_unit_cell: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct H2x2 {
    h: [[f32; 2]; 4], // 2x2 complex
}

// DIM: 2x2 h(k) -> 2 eigenvalues
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EigenInfo {
    k_value: [f32; 2],            // for mem align padding purpose
    eigenvalues: [f32; 2],            // two real eigenvalues
    eigenvectors: [[[f32; 2]; 2]; 2], // two complex 2-dim eigenvectors
}

fn calc_k_grid(context: &WgpuContext) {
    let mut command_encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.k_grid_pipeline);
        compute_pass.set_bind_group(0, &context.bind_group, &[]);
        compute_pass.dispatch_workgroups(32, 32, 1);
    }
    // We finish the compute pass by dropping it.

    // Finalize the command encoder, add the contained commands to the queue and flush.
    context.queue.submit(Some(command_encoder.finish()));

    context.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
}

fn calc_initial_eigen(context: &WgpuContext) {
    let mut command_encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.initial_eigen_pipeline);
        compute_pass.set_bind_group(0, &context.bind_group, &[]);
        compute_pass.dispatch_workgroups(32, 32, 1);
    }
    // We finish the compute pass by dropping it.

    // Finalize the command encoder, add the contained commands to the queue and flush.
    context.queue.submit(Some(command_encoder.finish()));

    context.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
}

async fn get_charge_density(context: &WgpuContext, chem_potential: f32) -> Vec<f32> {
    let mut command_encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    context.chem_potential_guess_buf_info.set_uniform_buffer(
        &context.device,
        &mut command_encoder,
        &[chem_potential],
    ).await.unwrap();

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.calc_charge_density_pipeline);
        compute_pass.set_bind_group(0, &context.bind_group, &[]);
        compute_pass.dispatch_workgroups(32, 32, 1);
    }
    // We finish the compute pass by dropping it.
    context
        .charge_density_buf_info
        .copy_to_staging_buffer(&mut command_encoder);

    // Finalize the command encoder, add the contained commands to the queue and flush.
    context.queue.submit(Some(command_encoder.finish()));

    let density = context
        .charge_density_buf_info
        .read_staging_buffer(&context.device)
        .await
        .unwrap();

    density
}

/// A convenient way to hold together all the useful wgpu stuff together.
struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    k_grid_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    k_values_buf_info: BufferInfo<{ 512 * 512 }, [f32; 2]>,
    hamiltonian_buf_info: BufferInfo<{ 512 * 512 }, H2x2>,
    energy_eigen_buf_info: BufferInfo<{ 512 * 512 }, EigenInfo>,
    chem_potential_guess_buf_info: BufferInfo<1, f32>,
    charge_density_buf_info: BufferInfo<{ 512 * 512 }, f32>,

    initial_eigen_pipeline: wgpu::ComputePipeline,
    calc_charge_density_pipeline: wgpu::ComputePipeline,
}

impl WgpuContext {
    async fn new() -> WgpuContext {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        // Our shader, kindly compiled with Naga.
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        use wgpu::BufferUsages;
        let system_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("System Info Buffer"),
            contents: bytemuck::cast_slice(&[*SYSTEM_INFO]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let k_values_buf_info = BufferInfo::<{ 512 * 512 }, [f32; 2]>::new(
            &device,
            Some("K Values"),
            1,
            BufferUsages::STORAGE,
        );

        let hamiltonian_buf_info = BufferInfo::<{ 512 * 512 }, H2x2>::new(
            &device,
            Some("Hamiltonian"),
            2,
            BufferUsages::STORAGE,
        );

        let energy_eigen_buf_info = BufferInfo::<{ 512 * 512 }, EigenInfo>::new(
            &device,
            Some("Eigen Info"),
            3,
            BufferUsages::STORAGE,
        );

        let chem_potential_guess_buf_info = BufferInfo::<1, f32>::new(
            &device,
            Some("Chem Potential Guess"),
            4,
            BufferUsages::UNIFORM,
        );

        let charge_density_buf_info = BufferInfo::<{ 512 * 512 }, f32>::new(
            &device,
            Some("Result Charge Density"),
            5,
            BufferUsages::STORAGE,
        );

        // This can be though of as the function signature for our CPU-GPU function.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // system_info buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        // Going to have this be None just to be safe.
                        min_binding_size: None,
                    },
                    count: None,
                },
                // k_values_buffer
                k_values_buf_info.get_bind_group_layout_entry(),
                // hamiltonian buffer
                hamiltonian_buf_info.get_bind_group_layout_entry(),
                // eigen_info buffer
                energy_eigen_buf_info.get_bind_group_layout_entry(),
                // chem optential guess buffer
                chem_potential_guess_buf_info.get_bind_group_layout_entry(),
                // charge density Buffer
                charge_density_buf_info.get_bind_group_layout_entry(),
            ],
        });
        // This ties actual resources stored in the GPU to our metaphorical function
        // through the binding slots we defined above.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: system_info_buffer.as_entire_binding(),
                },
                k_values_buf_info.get_bind_group_entry(),
                hamiltonian_buf_info.get_bind_group_entry(),
                energy_eigen_buf_info.get_bind_group_entry(),
                chem_potential_guess_buf_info.get_bind_group_entry(),
                charge_density_buf_info.get_bind_group_entry(),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let k_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_k_grid"),
            compilation_options: Default::default(),
            cache: None,
        });
        let initial_eigen_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("calc_initial_eigen"),
                compilation_options: Default::default(),
                cache: None,
            });

        let calc_charge_density_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("calc_charge_density"),
                compilation_options: Default::default(),
                cache: None,
            });

        WgpuContext {
            device,
            queue,
            k_values_buf_info,
            hamiltonian_buf_info,
            energy_eigen_buf_info,
            chem_potential_guess_buf_info,
            charge_density_buf_info,

            k_grid_pipeline,
            initial_eigen_pipeline,
            calc_charge_density_pipeline,
            bind_group,
        }
    }
}
