# AII Migration: Hardware Dispatch
## Document 4: Multi-Hardware Acceleration

### Hardware Dispatcher Core

**File:** `lib/aii/hardware_dispatcher.ex`

```elixir
defmodule AII.HardwareDispatcher do
  @moduledoc """
  Analyzes interaction AST and selects optimal hardware accelerator.
  Maps physics operations to specialized compute units.
  """
  
  @type hardware :: :auto | :rt_cores | :tensor_cores | :npu | :cuda_cores | :gpu | :cpu | :parallel | :simd
  
  def dispatch(interaction) do
    cond do
      has_spatial_query?(interaction) -> :rt_cores
      has_matrix_op?(interaction) -> :tensor_cores
      has_learned_model?(interaction) -> :npu
      has_parallel_compute?(interaction) -> :cuda_cores
      has_general_gpu?(interaction) -> :gpu
      has_multi_core_cpu?(interaction) -> :parallel
      has_vector_ops?(interaction) -> :simd
      true -> :cpu
    end
  end
  
  # RT Cores: BVH traversal, collision detection
  defp has_spatial_query?(interaction) do
    ast_contains?(interaction.body, [
      :nearby, :colliding?, :within_radius,
      :find_neighbors, :spatial_query, :ray_cast
    ])
  end
  
  # Tensor Cores: Matrix multiply, linear algebra
  defp has_matrix_op?(interaction) do
    ast_contains?(interaction.body, [
      :matrix_multiply, :dot_product, :matmul,
      :tensor_op, :outer_product, :linear_transform
    ])
  end
  
  # NPU: Neural network inference
  defp has_learned_model?(interaction) do
    ast_contains?(interaction.body, [
      :predict, :infer, :neural_network,
      :forward_pass, :model_eval
    ])
  end
  
  # CUDA: General parallel computation
  defp has_parallel_compute?(interaction) do
    ast_contains?(interaction.body, [
      :parallel_map, :reduce, :scan
    ])
  end
  
  # GPU: Vendor-agnostic GPU compute
  defp has_general_gpu?(interaction) do
    ast_contains?(interaction.body, [
      :gpu_compute, :shader, :compute_shader
    ])
  end
  
  # Multi-core CPU: Embarrassingly parallel
  defp has_multi_core_cpu?(interaction) do
    ast_contains?(interaction.body, [
      :flow_map, :task_async, :parallel_stream
    ])
  end
  
  # SIMD: Vector operations
  defp has_vector_ops?(interaction) do
    ast_contains?(interaction.body, [
      :vector_add, :vector_mul, :simd_map, :vectorized
    ])
  end
  
  defp ast_contains?(ast, keywords) do
    ast
    |> Macro.prewalk(false, fn
      {keyword, _, _}, _acc when keyword in keywords -> {keyword, true}
      node, acc -> {node, acc}
    end)
    |> elem(1)
  end
end
```

---

### Code Generation Strategy

**File:** `lib/aii/codegen.ex`

```elixir
defmodule AII.Codegen do
  alias AII.HardwareDispatcher
  
  def generate(interaction) do
    hardware = HardwareDispatcher.dispatch(interaction)
    
    case hardware do
      :auto -> generate_auto_dispatch(interaction)
      :rt_cores -> generate_ray_tracing(interaction)
      :tensor_cores -> generate_tensor_op(interaction)
      :npu -> generate_npu_inference(interaction)
      :cuda_cores -> generate_cuda_kernel(interaction)
      :gpu -> generate_generic_gpu(interaction)
      :parallel -> generate_multi_core_cpu(interaction)
      :simd -> generate_simd_code(interaction)
      :cpu -> generate_zig_code(interaction)
    end
  end
  
  # Generate Vulkan ray query for RT Cores
  defp generate_ray_tracing(interaction) do
    """
    // Vulkan Ray Tracing (RT Cores)
    VkAccelerationStructureKHR as = buildBVH(particles);
    
    for (int i = 0; i < numParticles; i++) {
        VkRayQueryKHR query;
        vkCmdInitRayQuery(
            query,
            as,
            particles[i].position,
            particles[i].radius * 2.0
        );
        
        // RT cores handle traversal in hardware
        while (vkRayQueryProceed(query)) {
            if (vkRayQueryGetIntersectionType(query) == VK_RAY_QUERY_INTERSECTION_TRIANGLE) {
                int hitIndex = vkRayQueryGetIntersectionInstanceId(query);
                // Process collision with particles[hitIndex]
            }
        }
    }
    """
  end
  
  # Generate GLSL compute shader for Tensor Cores
  defp generate_tensor_op(interaction) do
    """
    #version 450
    #extension GL_KHR_cooperative_matrix : enable
    
    layout(local_size_x = 16, local_size_y = 16) in;
    
    layout(binding = 0) buffer Positions {
        vec3 positions[];
    };
    
    layout(binding = 1) buffer Forces {
        vec3 forces[];
    };
    
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        
        // Use cooperative matrix (Tensor Cores!)
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> result;
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
        
        // Load data
        coopMatLoad(matA, positions, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matB, forces, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        
        // Tensor cores do matrix multiply
        result = coopMatMulAdd(matA, matB, result);
        
        // Store result
        coopMatStore(result, positions, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    }
    """
    # Generate auto-dispatch code (compiler chooses)
    defp generate_auto_dispatch(interaction) do
      # Analyze interaction and dispatch to optimal hardware
      # Includes fallback chain generation
      """
      // Auto-dispatch: Compiler will analyze and select optimal hardware
      // Fallback chain: RT Cores → Tensor Cores → NPU → CUDA → GPU → Parallel → SIMD → CPU
      analyze_and_dispatch(#{inspect(interaction)})
      """
    end
    
    # Generate vendor-agnostic GPU code
    defp generate_generic_gpu(interaction) do
      """
      // Vendor-agnostic GPU (Vulkan/OpenCL/Metal)
      #version 450
      layout(local_size_x = 256) in;
      
      void main() {
          uint idx = gl_GlobalInvocationID.x;
          // Generated GPU code
      }
      """
    end
    
    # Generate multi-core CPU code
    defp generate_multi_core_cpu(interaction) do
      """
      // Multi-core CPU using Elixir Flow/Task
      Flow.from_enumerable(data)
      |> Flow.partition()
      |> Flow.map(fn item ->
          // Parallel computation
      end)
      |> Enum.to_list()
      """
    end
    
    # Generate SIMD vectorized code
    defp generate_simd_code(interaction) do
      """
      // SIMD instructions (AVX2/AVX-512/NEON)
      simd_vector_add(
          vector_a,
          vector_b,
          vector_length
      )
      """
    end
  end
end
```

---

### Vulkan Backend (Zig)

**File:** `runtime/zig/vulkan_backend.zig`

```zig
const std = @import("std");
const vk = @import("vulkan");

pub const VulkanBackend = struct {
    instance: vk.Instance,
    device: vk.Device,
    queue: vk.Queue,
    
    // RT Cores support
    rt_enabled: bool,
    acceleration_structure: ?vk.AccelerationStructureKHR,
    
    // Tensor Cores support
    tensor_enabled: bool,
    
    pub fn init(allocator: std.mem.Allocator) !VulkanBackend {
        // Create Vulkan instance
        const app_info = vk.ApplicationInfo{
            .pApplicationName = "AII Runtime",
            .applicationVersion = vk.makeApiVersion(0, 1, 0, 0),
            .pEngineName = "BrokenRecord",
            .engineVersion = vk.makeApiVersion(0, 1, 0, 0),
            .apiVersion = vk.API_VERSION_1_3,
        };
        
        const instance = try vk.createInstance(&.{
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
        }, null);
        
        // Select device with RT/Tensor support
        const physical_device = try selectDevice(instance);
        
        // Check for ray tracing support
        const rt_enabled = hasRayTracing(physical_device);
        
        // Check for cooperative matrix (Tensor Core) support
        const tensor_enabled = hasCooperativeMatrix(physical_device);
        
        const device = try createDevice(physical_device, rt_enabled, tensor_enabled);
        const queue = vk.getDeviceQueue(device, 0, 0);
        
        return VulkanBackend{
            .instance = instance,
            .device = device,
            .queue = queue,
            .rt_enabled = rt_enabled,
            .tensor_enabled = tensor_enabled,
            .acceleration_structure = null,
        };
    }
    
    // Build BVH for RT Cores
    pub fn buildAccelerationStructure(
        self: *VulkanBackend,
        particles: []const Particle
    ) !void {
        if (!self.rt_enabled) return error.RayTracingNotSupported;
        
        // Create BLAS (Bottom Level Acceleration Structure)
        var geometries = try std.ArrayList(vk.AccelerationStructureGeometryKHR)
            .initCapacity(allocator, particles.len);
        
        for (particles) |p| {
            try geometries.append(.{
                .geometryType = .TRIANGLES_KHR,
                .geometry = .{
                    .triangles = .{
                        .vertexFormat = .R32G32B32_SFLOAT,
                        .vertexData = // particle as sphere mesh
                    }
                }
            });
        }
        
        const as = try vk.createAccelerationStructureKHR(
            self.device,
            &.{
                .type = .BOTTOM_LEVEL_KHR,
                .geometries = geometries.items,
            },
            null
        );
        
        self.acceleration_structure = as;
    }
    
    // Execute ray query (RT Cores)
    pub fn executeRayQuery(
        self: *VulkanBackend,
        origin: Vec3,
        radius: f32
    ) ![]u32 {
        const cmd_buffer = try allocateCommandBuffer(self.device);
        
        // Bind ray query pipeline
        vk.cmdBindPipeline(cmd_buffer, .RAY_TRACING_KHR, ray_query_pipeline);
        
        // Launch rays (RT cores execute this)
        vk.cmdTraceRaysKHR(
            cmd_buffer,
            &raygen_region,
            &miss_region,
            &hit_region,
            &callable_region,
            width,
            height,
            depth
        );
        
        // Submit and wait
        try submitAndWait(self.queue, cmd_buffer);
        
        // Read results
        return readResults(result_buffer);
    }
    
    // Execute tensor operation (Tensor Cores)
    pub fn executeTensorOp(
        self: *VulkanBackend,
        matA: []const f32,
        matB: []const f32
    ) ![]f32 {
        if (!self.tensor_enabled) return error.TensorCoresNotSupported;
        
        const cmd_buffer = try allocateCommandBuffer(self.device);
        
        // Bind compute pipeline with cooperative matrix
        vk.cmdBindPipeline(cmd_buffer, .COMPUTE, tensor_pipeline);
        
        // Bind buffers
        vk.cmdBindDescriptorSets(cmd_buffer, .COMPUTE, layout, 0, &[_]vk.DescriptorSet{desc_set}, &[_]u32{});
        
        // Dispatch (Tensor cores execute matrix multiply)
        vk.cmdDispatch(cmd_buffer, workgroup_x, workgroup_y, workgroup_z);
        
        try submitAndWait(self.queue, cmd_buffer);
        
        return readResults(result_buffer);
    }
};
```

---

### NPU Backend

**File:** `runtime/zig/npu_backend.zig`

```zig
const std = @import("std");

pub const NPUBackend = struct {
    platform: Platform,
    model: ?Model,
    
    pub const Platform = enum {
        apple_ane,      // Apple Neural Engine
        qualcomm_snpe,  // Qualcomm
        intel_openvino, // Intel
        none,
    };
    
    pub fn init() !NPUBackend {
        const platform = detectPlatform();
        return NPUBackend{
            .platform = platform,
            .model = null,
        };
    }
    
    fn detectPlatform() Platform {
        // Check what NPU is available
        if (std.Target.current.os.tag == .macos) {
            return .apple_ane;
        } else if (hasQualcommNPU()) {
            return .qualcomm_snpe;
        } else if (hasIntelNPU()) {
            return .intel_openvino;
        } else {
            return .none;
        }
    }
    
    pub fn loadModel(self: *NPUBackend, path: []const u8) !void {
        switch (self.platform) {
            .apple_ane => try loadCoreMLModel(path),
            .qualcomm_snpe => try loadSNPEModel(path),
            .intel_openvino => try loadOpenVINOModel(path),
            .none => return error.NPUNotAvailable,
        }
    }
    
    pub fn infer(
        self: *NPUBackend,
        input: []const f32
    ) ![]f32 {
        // Run inference on NPU
        switch (self.platform) {
            .apple_ane => return inferAppleANE(input),
            .qualcomm_snpe => return inferQualcomm(input),
            .intel_openvino => return inferIntel(input),
            .none => return error.NPUNotAvailable,
        }
    }
};
```

---

### Automatic Selection Example

**User writes this:**

```elixir
definteraction :find_nearby do
  let particle do
    # Compiler sees "nearby" keyword
    nearby = nearby_particles(particle.position, radius: 2.0)
    
    # Hardware dispatcher chooses RT Cores!
    # Generates Vulkan ray query automatically
  end
end
```

**Compiler generates:**

```zig
// Generated Zig code
fn interaction_find_nearby(
    particle: *Particle,
    all_particles: []Particle,
    backend: *VulkanBackend
) ![]u32 {
    // Use RT Cores for spatial query
    return try backend.executeRayQuery(
        particle.position,
        2.0  // radius
    );
}
```

---

### Performance Testing

**File:** `lib/aii/benchmark.ex`

```elixir
defmodule AII.Benchmark do
  def compare_hardware(interaction, particles) do
    # CPU
    {time_cpu, _} = :timer.tc(fn ->
      execute_on_cpu(interaction, particles)
    end)
    
    # CUDA
    {time_cuda, _} = :timer.tc(fn ->
      execute_on_cuda(interaction, particles)
    end)
    
    # RT Cores
    {time_rt, _} = :timer.tc(fn ->
      execute_on_rt_cores(interaction, particles)
    end)
    
    # Tensor Cores
    {time_tensor, _} = :timer.tc(fn ->
      execute_on_tensor_cores(interaction, particles)
    end)
    
    %{
      cpu: time_cpu / 1000,  # Convert to ms
      cuda: time_cuda / 1000,
      rt_cores: time_rt / 1000,
      tensor_cores: time_tensor / 1000,
      speedup: %{
        cuda: time_cpu / time_cuda,
        rt_cores: time_cpu / time_rt,
        tensor_cores: time_cpu / time_tensor
      }
    }
  end
end
```

---

### Key Points

**1. Automatic Dispatch**
- Compiler analyzes AST keywords
- Selects optimal hardware
- No manual kernel writing

**2. Hardware Abstraction**
- Same DSL code
- Different backends
- Transparent acceleration

**3. Fallback Strategy**
```
RT Cores available? → Use RT Cores
    ↓ No
Tensor Cores available? → Use Tensor Cores
    ↓ No
NPU available? → Use NPU
    ↓ No
CUDA available? → Use CUDA
    ↓ No
GPU available? → Use Generic GPU
    ↓ No
Multi-core CPU? → Use Parallel
    ↓ No
SIMD available? → Use SIMD
    ↓ No
CPU fallback
```

**4. Detection**
```zig
// Query available hardware
const caps = queryHardwareCapabilities();

if (caps.rt_cores) {
    // Use ray tracing
} else if (caps.tensor_cores) {
    // Use tensor ops
} else {
    // CPU fallback
}
```

---

### Next Steps

1. Implement comprehensive hardware detection (Vulkan/OpenCL/Metal/SIMD)
2. Build RT Core backend (ray queries)
3. Build Tensor Core backend (cooperative matrix)
4. Build NPU platform detection (Apple/AMD/Intel/Qualcomm)
5. Build generic GPU backend (vendor-agnostic)
6. Build multi-core CPU backend (Flow/Task)
7. Build SIMD backend (AVX/NEON)
8. Implement auto-dispatch with fallback chains
9. Automatic code generation for all backends
10. Benchmark all backends with platform-specific optimizations
