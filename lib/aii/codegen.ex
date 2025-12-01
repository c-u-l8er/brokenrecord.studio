defmodule AII.Codegen do
  @moduledoc """
  Code generation for hardware-specific execution of AII interactions.

  Generates optimized code for different accelerators:
  - RT Cores: Vulkan ray tracing
  - Tensor Cores: Cooperative matrix operations
  - NPU: Neural inference
  - CUDA: General GPU compute
  - GPU: Vendor-agnostic GPU
  - Parallel: Multi-core CPU
  - SIMD: Vectorized CPU
  - CPU: Fallback CPU
  """

  require Logger

  use Agent

  alias AII.HardwareDispatcher

  @type hardware :: HardwareDispatcher.hardware()
  @type interaction :: map()
  @type generated_code :: String.t() | binary()

  # Start the cache agent
  def start_link(_opts \\ []) do
    Agent.start_link(fn -> %{} end, name: __MODULE__)
  end

  @doc """
  Generates code for the given interaction on the specified hardware.

  ## Parameters
  - `interaction`: The interaction AST/map to generate code for
  - `hardware`: The target hardware accelerator

  ## Returns
  - Generated code as a string
  """
  @spec generate(interaction(), hardware()) :: generated_code()
  def generate(interaction, hardware) do
    generate_uncached(interaction, hardware)
  end

  @doc """
  Clear the code generation cache.
  """
  @spec clear_cache() :: :ok
  def clear_cache do
    Agent.update(__MODULE__, fn _ -> %{} end)
  end

  # Private functions

  @spec generate_uncached(interaction(), hardware()) :: generated_code()
  defp generate_uncached(interaction, hardware) do
    case hardware do
      :rt_cores -> generate_rt_cores(interaction)
      :tensor_cores -> generate_tensor_cores(interaction, hardware)
      :npu -> generate_npu(interaction)
      :cuda -> generate_cuda_cores(interaction)
      :cuda_cores -> generate_cuda_cores(interaction)
      # Now returns SPIR-V binary
      :gpu -> generate_gpu(interaction)
      :parallel -> generate_parallel(interaction)
      :simd -> generate_simd(interaction)
      :cpu -> generate_cpu(interaction)
      :auto -> generate_auto(interaction)
      # Fallback for unknown hardware
      _ -> generate_cpu(interaction)
    end
  end

  @spec get_cached_code(term()) :: {:ok, generated_code()} | :not_found
  defp get_cached_code(cache_key) do
    try do
      case Agent.get(__MODULE__, &Map.get(&1, cache_key)) do
        nil -> :not_found
        code -> {:ok, code}
      end
    catch
      # Agent not started
      :exit, _ -> :not_found
    end
  end

  @spec cache_code(term(), generated_code()) :: :ok
  defp cache_code(cache_key, code) do
    try do
      Agent.update(__MODULE__, &Map.put(&1, cache_key, code))
    catch
      # Agent not started, skip caching
      :exit, _ -> :ok
    end
  end

  @doc """
  Generates code with automatic hardware selection.
  """
  @spec generate_auto(interaction()) :: generated_code()
  def generate_auto(interaction) do
    case HardwareDispatcher.dispatch(interaction, :auto) do
      {:ok, hardware} -> generate(interaction, hardware)
      {:error, _} -> generate_cpu(interaction)
    end
  end

  # RT Cores: Vulkan Ray Tracing
  defp generate_rt_cores(interaction) do
    """
    // Vulkan Ray Tracing (RT Cores) - Generated for #{inspect(interaction)}

    // Build acceleration structure
    VkAccelerationStructureKHR as = buildBVH(particles);

    // Ray query shader
    #version 460
    #extension GL_EXT_ray_query : require

    void main() {
        uint idx = gl_GlobalInvocationID.x;

        // Initialize ray query
        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, as, gl_RayFlagsTerminateOnFirstHitEXT,
                              0xFF, origin, 0.0, direction, far_plane);

        // Execute ray query (RT cores accelerate this)
        while (rayQueryProceedEXT(rayQuery)) {
            if (rayQueryGetIntersectionTypeEXT(rayQuery) == gl_RayQueryCommittedIntersectionTriangleEXT) {
                // Process collision/interaction
                process_hit(rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery));
            }
        }
    }
    """
  end

  # Tensor Cores: Cooperative Matrix or WMMA
  defp generate_tensor_cores(interaction, hardware) do
    case hardware do
      :cuda_tensor_cores ->
        generate_cuda_tensor_cores(interaction)

      _ ->
        """
        // Vulkan Tensor Cores - Generated for #{inspect(interaction)}

        #version 450
        #extension GL_KHR_cooperative_matrix : enable

        layout(local_size_x = 16, local_size_y = 16) in;

        // Cooperative matrix declarations
        layout(binding = 0) buffer Matrices {
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matrixA[];
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matrixB[];
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> result[];
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;

            // Load matrices
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a = matrixA[idx];
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b = matrixB[idx];
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> r = result[idx];

            // Tensor cores execute this multiply-accumulate
            r = coopMatMulAdd(a, b, r);

            // Store result
            result[idx] = r;
        }
        """
    end
  end

  # NPU: Neural Processing Unit
  defp generate_npu(interaction) do
    """
    // NPU Inference Code - Generated for #{inspect(interaction)}

    // Platform-specific NPU code
    #ifdef __APPLE__
    // Apple Neural Engine
    @interface AIIInference : NSObject
    - (MLMultiArray *)runInference:(MLMultiArray *)input;
    @end

    @implementation AIIInference
    - (MLMultiArray *)runInference:(MLMultiArray *)input {
        // Load Core ML model
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:&error];

        // Create prediction
        MLDictionaryFeatureProvider *inputProvider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"input": input}
                                                              error:&error];

        id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider error:&error];

        return output[@"output"];
    }
    @end

    #elif defined(__ANDROID__)
    // Android NNAPI
    ANeuralNetworksModel *model = nullptr;
    ANeuralNetworksCompilation *compilation = nullptr;

    // Create model and add operations
    ANeuralNetworksModel_create(&model);
    // ... add operations for #{inspect(interaction)}

    // Compile for NPU
    ANeuralNetworksCompilation_create(model, &compilation);
    ANeuralNetworksCompilation_finish(compilation);

    // Execute
    ANeuralNetworksExecution *execution = nullptr;
    ANeuralNetworksExecution_create(compilation, &execution);
    // ... set inputs and run

    #else
    // Generic NPU fallback
    // Use OpenVINO, TVM, or similar
    #endif
    """
  end

  # CUDA Cores: General GPU Compute
  defp generate_cuda_cores(interaction) do
    """
    // CUDA Kernel - Generated for #{inspect(interaction)}

    __global__ void aii_kernel(float *particles, int num_particles) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < num_particles) {
            // Generated CUDA code for interaction
            float3 position = make_float3(
                particles[idx * 6 + 0],
                particles[idx * 6 + 1],
                particles[idx * 6 + 2]
            );

            float3 velocity = make_float3(
                particles[idx * 6 + 3],
                particles[idx * 6 + 4],
                particles[idx * 6 + 5]
            );

            // Apply interaction (example: gravity)
            velocity.y -= 9.81f * 0.016f;  // dt = 1/60

            // Update position
            position.x += velocity.x * 0.016f;
            position.y += velocity.y * 0.016f;
            position.z += velocity.z * 0.016f;

            // Write back
            particles[idx * 6 + 0] = position.x;
            particles[idx * 6 + 1] = position.y;
            particles[idx * 6 + 2] = position.z;
            particles[idx * 6 + 3] = velocity.x;
            particles[idx * 6 + 4] = velocity.y;
            particles[idx * 6 + 5] = velocity.z;
        }
    }

    // Host code
    extern "C" void run_aii_kernel(float *particles, int num_particles) {
        float *d_particles;
        cudaMalloc(&d_particles, num_particles * 6 * sizeof(float));
        cudaMemcpy(d_particles, particles, num_particles * 6 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(256);
        dim3 grid((num_particles + block.x - 1) / block.x);

        aii_kernel<<<grid, block>>>(d_particles, num_particles);

        cudaMemcpy(particles, d_particles, num_particles * 6 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_particles);
    }
    """
  end

  # CUDA Tensor Cores: WMMA Operations
  defp generate_cuda_tensor_cores(interaction) do
    """
    // CUDA Tensor Cores - Generated for #{inspect(interaction)}

    #include <mma.h>
    using namespace nvcuda::wmma;

    __global__ void tensor_core_matmul(
        half *a, half *b, float *c,
        int m, int n, int k
    ) {
        // Declare fragments
        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;

        // Calculate global warp indices
        int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        int warpN = blockIdx.y * blockDim.y + threadIdx.y;

        // Load matrices
        load_matrix_sync(a_frag, a + warpM * 16 * k, k);
        load_matrix_sync(b_frag, b + warpN * 16, k);
        load_matrix_sync(c_frag, c + warpM * 16 * n + warpN * 16, n, mem_row_major);

        // Perform matrix multiplication
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store result
        store_matrix_sync(c + warpM * 16 * n + warpN * 16, c_frag, n, mem_row_major);
    }

    __global__ void tensor_core_force_calc(
        float3 *positions, float *masses, float3 *forces,
        int num_particles
    ) {
        // Tensor core optimized N-body force calculation
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < num_particles) {
            float3 pos_i = positions[idx];
            float m_i = masses[idx];
            float3 force = make_float3(0.0f, 0.0f, 0.0f);

            // Use tensor cores for batched distance calculations
            // Simplified version - in practice would use WMMA for matrix ops
            for (int j = 0; j < num_particles; j++) {
                if (idx != j) {
                    float3 r = positions[j] - pos_i;
                    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z + 1e-10f);
                    float f = 6.67430e-11f * m_i * masses[j] / (dist * dist * dist);
                    force += f * r;
                }
            }

            forces[idx] = force;
        }
    }
    """
  end

  # GPU: General GPU Compute - Returns SPIR-V binary for particle physics
  defp generate_gpu(interaction) do
    glsl_source = generate_glsl_compute_shader(interaction)
    compile_glsl_to_spirv(glsl_source)
  end

  # Generate GLSL compute shader source
  defp generate_glsl_compute_shader(interaction) do
    # Extract interaction type
    interaction_type =
      case interaction do
        %{body: {type, _, _}} -> type
        %{name: name} -> name
        _ -> :default
      end

    # Generate shader based on interaction
    base_shader = """
    #version 450
    #extension GL_ARB_compute_shader : enable

    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    // Storage buffers for particle data
    layout(binding = 0) buffer Positions {
        vec3 positions[];
    };

    layout(binding = 1) buffer Velocities {
        vec3 velocities[];
    };

    layout(binding = 2) buffer Masses {
        float masses[];
    };

    // Uniform buffer for simulation parameters
    layout(binding = 3) uniform SimulationParams {
        float dt;
        float gravity;
        uint num_particles;
    };

    void main() {
        uint idx = gl_GlobalInvocationID.x;

        // Bounds check
        if (idx >= num_particles) {
            return;
        }

        // Load particle data
        vec3 position = positions[idx];
        vec3 velocity = velocities[idx];
        float mass = masses[idx];
    """

    # Add interaction-specific logic
    specific_logic =
      case interaction_type do
        :gravity ->
          """
          // Apply gravity force
          vec3 force = vec3(0.0, -gravity * mass, 0.0);
          """

        :collision ->
          """
          // Collision detection and response
          vec3 force = vec3(0.0, 0.0, 0.0);
          // Simple collision with ground plane
          if (position.y < 0.0) {
            velocity.y = -velocity.y * 0.8; // Bounce with damping
            position.y = 0.0;
          }
          """

        _ ->
          """
          // Default: gravity
          vec3 force = vec3(0.0, -gravity * mass, 0.0);
          """
      end

    integration = """
        // Integrate velocity (F = ma, a = F/m)
        vec3 acceleration = force / mass;
        velocity += acceleration * dt;

        // Integrate position
        position += velocity * dt;

        // Store updated data
        positions[idx] = position;
        velocities[idx] = velocity;
    """

    base_shader <> specific_logic <> integration <> "\n    }\n"
  end

  # Compile GLSL source to SPIR-V binary
  defp compile_glsl_to_spirv(glsl_source) do
    # Create temporary files
    tmp_dir = System.tmp_dir!()
    glsl_file = Path.join(tmp_dir, "aii_shader_#{:rand.uniform(1_000_000)}.comp")
    spirv_file = Path.join(tmp_dir, "aii_shader_#{:rand.uniform(1_000_000)}.spv")

    try do
      # Write GLSL to temp file
      File.write!(glsl_file, glsl_source)

      # Run glslangValidator
      {output, exit_code} =
        System.cmd("glslangValidator", ["-V", "-o", spirv_file, glsl_file],
          stderr_to_stdout: true
        )

      if exit_code != 0 do
        Logger.error("GLSL compilation failed: #{output}")
        raise "Failed to compile GLSL to SPIR-V"
      end

      # Read SPIR-V binary
      File.read!(spirv_file)
    after
      # Clean up temp files
      File.rm(glsl_file)
      File.rm(spirv_file)
    end
  end

  # Multi-core CPU: Parallel Elixir
  defp generate_parallel(interaction) do
    """
    # Elixir Parallel Code - Generated for #{inspect(interaction)}

    defmodule GeneratedParallel do
      def execute(particles) do
        # Use Flow for parallel processing
        particles
        |> Flow.from_enumerable()
        |> Flow.partition()
        |> Flow.map(&process_particle/1)
        |> Enum.to_list()
      end

      defp process_particle(particle) do
        # Generated particle processing code
        %{particle |
          velocity: %{
            x: particle.velocity.x,
            y: particle.velocity.y - 9.81 * 0.016,  # Gravity
            z: particle.velocity.z
          },
          position: %{
            x: particle.position.x + particle.velocity.x * 0.016,
            y: particle.position.y + particle.velocity.y * 0.016,
            z: particle.position.z + particle.velocity.z * 0.016
          }
        }
      end
    end
    """
  end

  # SIMD: Vectorized CPU
  defp generate_simd(interaction) do
    """
    // SIMD Vectorized Code - Generated for #{inspect(interaction)}

    #include <immintrin.h>  // AVX2/AVX-512

    void process_particles_simd(float *positions, float *velocities, size_t num_particles) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_particles; i += 8) {  // Process 8 particles at once
            // Load 8 x positions (x,y,z components)
            __m256 px = _mm256_load_ps(&positions[i * 3 + 0]);
            __m256 py = _mm256_load_ps(&positions[i * 3 + 1]);
            __m256 pz = _mm256_load_ps(&positions[i * 3 + 2]);

            // Load 8 x velocities
            __m256 vx = _mm256_load_ps(&velocities[i * 3 + 0]);
            __m256 vy = _mm256_load_ps(&velocities[i * 3 + 1]);
            __m256 vz = _mm256_load_ps(&velocities[i * 3 + 2]);

            // Apply gravity (vy -= 9.81 * dt)
            __m256 gravity = _mm256_set1_ps(-9.81f * 0.016f);
            vy = _mm256_add_ps(vy, gravity);

            // Update positions (p += v * dt)
            __m256 dt = _mm256_set1_ps(0.016f);
            px = _mm256_fmadd_ps(vx, dt, px);
            py = _mm256_fmadd_ps(vy, dt, py);
            pz = _mm256_fmadd_ps(vz, dt, pz);

            // Store results
            _mm256_store_ps(&positions[i * 3 + 0], px);
            _mm256_store_ps(&positions[i * 3 + 1], py);
            _mm256_store_ps(&positions[i * 3 + 2], pz);
            _mm256_store_ps(&velocities[i * 3 + 0], vx);
            _mm256_store_ps(&velocities[i * 3 + 1], vy);
            _mm256_store_ps(&velocities[i * 3 + 2], vz);
        }
    }
    """
  end

  # CPU: Fallback Zig/Elixir
  defp generate_cpu(interaction) do
    """
    // CPU Fallback Code - Generated for #{inspect(interaction)}

    // Zig implementation (would be compiled to NIF)
    pub fn process_particles_cpu(
        positions: []@Vector(3, f32),
        velocities: []@Vector(3, f32),
        masses: []f32
    ) void {
        for (positions, velocities, masses) |*pos, *vel, mass| {
            _ = mass;  // Not used in this example

            // Apply gravity
            vel[1] -= 9.81 * 0.016;

            // Update position
            pos[0] += vel[0] * 0.016;
            pos[1] += vel[1] * 0.016;
            pos[2] += vel[2] * 0.016;
        }
    }

    // Or Elixir implementation
    defmodule GeneratedCPU do
      def execute(particles) do
        Enum.map(particles, fn particle ->
          # Simple Euler integration with gravity
          dt = 0.016

          velocity = %{
            x: particle.velocity.x,
            y: particle.velocity.y - 9.81 * dt,
            z: particle.velocity.z
          }

          position = %{
            x: particle.position.x + velocity.x * dt,
            y: particle.position.y + velocity.y * dt,
            z: particle.position.z + velocity.z * dt
          }

          %{particle | position: position, velocity: velocity}
        end)
      end
    end
    """
  end
end
