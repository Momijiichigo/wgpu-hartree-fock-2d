pub struct BufferInfo<const NUM_T: u64, T: bytemuck::Pod> {
    usage: wgpu::BufferUsages,
    shader_buffer: wgpu::Buffer,
    shader_buffer_label: Option<String>,
    staging_buffer: wgpu::Buffer,
    staging_buffer_label: Option<String>,
    binding: u32,
    _marker: std::marker::PhantomData<T>,
}

impl<const NUM_T: u64, T: bytemuck::Pod> BufferInfo<NUM_T, T> {
    pub fn new(
        device: &wgpu::Device,
        label: Option<&str>,
        binding: u32,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let shader_buffer_label = label.map(|s| format!("{} Shader Buffer", s));

        // For convenience, I expect the user to provide just BufferUsages::STORAGE or
        // BufferUsages::UNIFORM.
        // I will generate the rest of the usages.
        let shader_buf_usage;
        let staging_buf_usage;
        if usage.contains(wgpu::BufferUsages::STORAGE) {
            shader_buf_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            staging_buf_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        } else if usage.contains(wgpu::BufferUsages::UNIFORM) {
            shader_buf_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
            staging_buf_usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE;
        } else {
            panic!("Invalid buffer usage");
        }

        let buff_size = NUM_T * std::mem::size_of::<T>() as u64;
        let shader_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: shader_buffer_label.as_deref(),
            size: buff_size,
            usage: shader_buf_usage,
            mapped_at_creation: false,
        });
        let staging_buffer_label = label.map(|s| format!("{} Staging Buffer", s));
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: staging_buffer_label.as_deref(),
            size: buff_size,
            usage: staging_buf_usage,
            mapped_at_creation: false,
        });
        Self {
            shader_buffer,
            shader_buffer_label,
            staging_buffer,
            staging_buffer_label,
            binding,
            usage,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn get_bind_group_layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        let binding_type = if self.usage.contains(wgpu::BufferUsages::STORAGE) {
            wgpu::BufferBindingType::Storage { read_only: false }
        } else if self.usage.contains(wgpu::BufferUsages::UNIFORM) {
            wgpu::BufferBindingType::Uniform
        } else {
            panic!("Invalid buffer usage");
        };
        wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: binding_type,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
    pub fn get_bind_group_entry(&self) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding: self.binding,
            resource: self.shader_buffer.as_entire_binding(),
        }
    }
    pub fn copy_to_staging_buffer(&self, command_encoder: &mut wgpu::CommandEncoder) {
        command_encoder.copy_buffer_to_buffer(
            &self.shader_buffer,
            0,
            &self.staging_buffer,
            0,
            NUM_T * std::mem::size_of::<T>() as u64,
        );
    }
    pub async fn set_uniform_buffer(
        &self,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
        data: &[T],
    ) -> Result<(), wgpu::BufferAsyncError> {
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Write, move |r| {
            sender.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver.recv_async().await.unwrap()?;
        {
            let mut view = buffer_slice.get_mapped_range_mut();
            view.copy_from_slice(bytemuck::cast_slice(data));
        }
        self.staging_buffer.unmap();

        command_encoder.copy_buffer_to_buffer(
            &self.staging_buffer,
            0,
            &self.shader_buffer,
            0,
            NUM_T * std::mem::size_of::<T>() as u64,
        );
        Ok(())
    }
    pub async fn read_staging_buffer(
        &self,
        device: &wgpu::Device,
    ) -> Result<Vec<T>, wgpu::BufferAsyncError> {
        let mut local_buffer = vec![T::zeroed(); NUM_T as usize];
        let (sender, receiver) = flume::bounded(1);
        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        receiver.recv_async().await.unwrap()?;
        {
            let view = buffer_slice.get_mapped_range();
            local_buffer.copy_from_slice(bytemuck::cast_slice(&view));
        }
        self.staging_buffer.unmap();

        Ok(local_buffer)
    }
}