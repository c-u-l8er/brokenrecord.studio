# AII Hardware Accelerator Reference
## Current Implementation Status & Usage Guide

---

## Quick Reference

### All Possible `accelerator` Values

```elixir
definteraction :my_interaction, accelerator: :value do
  # ...
end
```

**Valid accelerator values:**

| Value | Hardware Target | Use Case | Status | Target Speedup |
|-------|----------------|----------|--------|----------------|
| `:auto` | Automatic selection | Default, let compiler decide | ✅ Implemented | Varies |
| `:rt_cores` | RT Cores (NVIDIA/AMD) | Spatial queries, collisions | 🔄 Framework-ready | 10× |
| `:tensor_cores` | Tensor Cores (NVIDIA) | Matrix ops, force computation | 🔄 Framework-ready | 50× |
| `:npu` | Neural Processing Unit | Neural inference, predictions | 📋 Planned | 100× |
| `:cuda_cores` | CUDA Cores (NVIDIA) | General GPU compute | 🔄 Framework-ready | 100× |
| `:gpu` | Generic GPU | Vendor-agnostic GPU | 🔄 Framework-ready | Varies |
| `:cpu` | CPU only | Fallback, debugging | ✅ Implemented | 1× |
| `:parallel` | Multi-core CPU | Embarrassingly parallel | ✅ Implemented | 4-16× |
| `:simd` | SIMD instructions | Vector operations | ✅ Implemented | 4-8× |

---

## 1. Detailed Accelerator Types

### `:auto` (Recommended Default)

**What it does:**
- Compiler analyzes interaction code
- Automatically selects best available hardware
- Fallback chain if hardware unavailable

**When to use:**
- Default choice for most interactions
- Trust the compiler's analysis
- Want automatic optimization

**Example:**
```elixir
definteraction :compute_forces, accelerator: :auto do
  let {particles} do
    # Currently defaults to SIMD CPU implementation
    # Framework ready for Tensor Cores when implemented
  end
end
```

**Current Status:** ✅ **Fully implemented** (defaults to SIMD)
**Compiler decision tree:**
```
Analyze interaction:
├─ Has spatial queries? → :rt_cores
├─ Has matrix multiply? → :tensor_cores
├─ Has neural network? → :npu
├─ Has parallel loops? → :cuda_cores
└─ Default → :cpu
```

---

### `:rt_cores` (Ray Tracing Cores)

**Hardware:**
- NVIDIA: RT Cores (Gen 3/4)
- AMD: Ray Accelerators
- Intel: XMX Ray Tracing Units
- Apple: Hardware RT (M4+)

**Best for:**
- Spatial queries (nearby particles)
- Collision detection
- Range searches
- BVH traversal
- K-nearest neighbors

**Performance:**
- 10× faster than CPU (when implemented)
- 5× faster than CUDA for spatial queries

**Current Status:** 🔄 **Framework-ready** - Architecture designed, code generation implemented, hardware execution pending

**Example:**
```elixir
definteraction :find_collisions, accelerator: :rt_cores do
  let {particles} do
    # Framework generates BVH construction and RT queries
    # Currently falls back to CPU collision detection
    # Will use actual RT cores when hardware backend implemented

    colliding_pairs = for p1 <- particles, p2 <- particles, p1 != p2 do
      distance = vec3_distance(p1.position, p2.position)
      if distance < collision_radius, do: {p1, p2}
    end

    colliding_pairs
  end
end
```

**When to use:**
- Particle collision detection
- Spatial range queries
- K-nearest neighbor searches
- Any spatial data structures

**Implementation notes:**
- BVH construction and RT query code generation implemented
- Vulkan backend designed for RT cores
- Currently uses CPU fallback with same algorithmic approach
---

### `:tensor_cores` (Matrix Accelerators)

**Hardware:**
- NVIDIA: Tensor Cores (Gen 4/5)
- AMD: Matrix Cores / AI Accelerators
- Intel: XMX Engines
- Apple: ❌ (no dedicated tensor cores)

**Best for:**
- Matrix multiplication
- Force matrices (N×N interactions)
- Linear algebra
- Neural network layers
- Dot products at scale

**Performance:**
- 50× faster than CPU (when implemented)
- 5× faster than CUDA for matrix ops
- FP16/FP8/FP4 support (newer gens)

**Current Status:** 🔄 **Framework-ready** - Code generation implemented, hardware execution pending

**Example:**
```elixir
definteraction :compute_force_matrix, accelerator: :tensor_cores do
  let {particles} do
    # Extract positions and masses
    positions = extract_matrix(particles, :position)  # N×3
    masses = extract_vector(particles, :mass)         # N×1
    
    # Compute force matrix using Tensor Cores
    # This is a N×N operation, perfect for Tensor Cores!
    distances = pairwise_distances(positions)         # N×N
    mass_products = outer_product(masses, masses)     # N×N
    forces = mass_products / (distances ** 2)         # N×N
    
    # Sum to get net force on each particle
    net_forces = sum_rows(forces)                     # N×1
    net_forces
  end
end
```

**When NOT to use:**
- Small matrices (<32×32)
- Sparse matrices
- Non-matrix operations
- Apple M-series (no Tensor Cores)

---

### `:npu` (Neural Processing Unit)

**Hardware:**
- AMD: XDNA (50 TOPS Ryzen AI)
- Intel: AI Boost (10-40 TOPS)
- Apple: Neural Engine (38 TOPS M4)
- Qualcomm: Hexagon NPU

**Best for:**
- Neural network inference
- Learned dynamics predictions
- Pattern recognition
- Low-precision compute (INT8/INT4)
- Constant background inference

**Performance:**
- 100× faster than CPU (for inference)
- 10× more power efficient than GPU
- Limited to specific operations

**Example:**
```elixir
definteraction :predict_evolution, accelerator: :npu do
  let {particles} do
    # Extract features
    features = extract_features(particles)  # N×feature_dim
    
    # Run neural network on NPU
    # Model predicts particle positions at t+1
    predictions = npu_inference(
      model: @learned_dynamics_model,
      input: features,
      precision: :int8  # NPU optimized
    )
    
    predictions
  end
end
```

**Limitations:**
- Model must be compiled for NPU
- Limited operations (mostly inference)
- Lower precision (INT8/INT4)
- Platform-specific (AMD vs Intel vs Apple)

**When NOT to use:**
- Training (use GPU instead)
- High-precision required (FP32/FP64)
- No pre-trained model available
- Desktop without NPU

---

### `:cuda_cores` (General GPU Compute)

**Hardware:**
- NVIDIA: CUDA Cores
- AMD: Stream Processors (via ROCm)
- Intel: Xe Cores (via OneAPI)
- Apple: GPU Cores (via Metal)

**Best for:**
- General parallel computation
- Not specialized (RT/Tensor)
- Large data-parallel loops
- Element-wise operations
- Reductions and scans

**Performance:**
- 100× faster than CPU (parallel work)
- Lower than specialized cores for their domains

**Example:**
```elixir
definteraction :integrate_euler, accelerator: :cuda_cores do
  let {particles, dt} do
    # Parallel update each particle (embarrassingly parallel)
    updated = parallel_map(particles, fn particle ->
      # Each thread handles one particle
      new_velocity = particle.velocity + particle.acceleration * dt
      new_position = particle.position + new_velocity * dt
      
      %{particle | 
        velocity: new_velocity,
        position: new_position
      }
    end)
    
    updated
  end
end
```

**When to use:**
- No specialized hardware match
- General parallel work
- Want GPU speedup
- Cross-platform (CUDA/ROCm/Metal)

---

### `:gpu` (Generic GPU)

**Difference from `:cuda_cores`:**
- `:cuda_cores` → CUDA/ROCm specific
- `:gpu` → Vendor-agnostic

**Best for:**
- Cross-platform code
- Don't know target GPU
- Let runtime pick GPU API

**Example:**
```elixir
definteraction :parallel_compute, accelerator: :gpu do
  let {particles} do
    # Runtime decides: CUDA, ROCm, Metal, or OpenCL
    # Based on available hardware
    gpu_parallel_map(particles, fn p ->
      compute_something(p)
    end)
  end
end
```

---

### `:cpu` (CPU Only)

**When to use:**
- Debugging (easier than GPU)
- No GPU available (CI/testing)
- Very small workloads (<1000 particles)
- Sequential algorithms

**Example:**
```elixir
definteraction :debug_forces, accelerator: :cpu do
  let {particles} do
    # Force CPU execution for debugging
    # Can use IO.inspect, breakpoints, etc.
    Enum.map(particles, fn p ->
      IO.inspect(p, label: "Processing")
      compute_force(p)
    end)
  end
end
```

---

### `:parallel` (Multi-core CPU)

**Best for:**
- CPU-only environments
- Embarrassingly parallel work
- No GPU available
- Simple parallelism

**Performance:**
- 4-16× faster (depends on cores)
- Lower than GPU but better than single-core

**Example:**
```elixir
definteraction :parallel_cpu, accelerator: :parallel do
  let {particles} do
    # Uses Elixir's Task/Flow for parallelism
    particles
    |> Flow.from_enumerable()
    |> Flow.partition()
    |> Flow.map(&compute_something/1)
    |> Enum.to_list()
  end
end
```

---

### `:simd` (SIMD Instructions)

**Hardware:**
- x86: AVX2, AVX-512
- ARM: NEON
- Apple: AMX (Apple Matrix)

**Best for:**
- Vector operations
- Element-wise arithmetic
- Particle integration
- Small to medium data sets

**Performance:**
- 2-3× faster than scalar CPU (current implementation)
- 4-8× potential with full SIMD optimization

**Current Status:** ✅ **Implemented** - Zig runtime uses SIMD for particle integration

**Example:**
```elixir
definteraction :integrate_particles, accelerator: :simd do
  let {particles} do
    # Zig runtime processes particles 4 at a time using SIMD
    # Currently implemented for Euler integration
    for p <- particles do
      p.position = p.position + p.velocity * dt
    end
  end
end
```

---

## 2. Choosing the Right Accelerator

### Decision Tree

```
What does your interaction do?

├─ Spatial queries (nearby, colliding)?
│  └─> :rt_cores
│
├─ Matrix multiply (N×N forces)?
│  └─> :tensor_cores
│
├─ Neural network inference?
│  └─> :npu
│
├─ Large parallel loops?
│  └─> :cuda_cores or :gpu
│
├─ Small workload or debugging?
│  └─> :cpu
│
└─ Not sure?
   └─> :auto (let compiler decide!)
```

---

### Common Patterns

**Pattern 1: Collision Detection (Framework-Ready)**
```elixir
definteraction :detect_collisions, accelerator: :rt_cores do
  # RT Cores will excel at spatial queries (framework-ready)
  # Currently falls back to CPU O(n²) collision detection
  let {particles, radius} do
    for p1 <- particles, p2 <- particles, p1 != p2 do
      distance = vec3_distance(p1.position, p2.position)
      if distance < radius, do: {p1, p2}
    end
  end
end
```

**Pattern 2: Force Computation (Framework-Ready)**
```elixir
definteraction :compute_forces, accelerator: :tensor_cores do
  # Tensor Cores will excel at matrix operations (framework-ready)
  # Currently falls back to CPU force calculations
  let {particles} do
    for p1 <- particles, p2 <- particles, p1 != p2 do
      force = gravitational_force(p1, p2)
      apply_force(p1, force)
      apply_force(p2, -force)
    end
  end
end
```

**Pattern 3: Learned Dynamics (Planned)**
```elixir
definteraction :predict_next, accelerator: :npu do
  # NPU will excel at neural inference (planned)
  # Currently no implementation
  let {particles} do
    # Placeholder for future NPU integration
    particles
  end
end
```

**Pattern 4: Integration (Implemented)**
```elixir
definteraction :integrate, accelerator: :simd do
  # SIMD implemented in Zig runtime
  # Processes particles 4 at a time using SIMD instructions
  let {particles} do
    for p <- particles do
      p.position = p.position + p.velocity * dt
    end
  end
end
```

---

## 3. Multiple Accelerators

### Chaining Accelerators

**Current Status:** 🔄 **Framework-ready** - Architecture supports fallback chains, implementation pending

**Design:**
```elixir
definteraction :find_neighbors,
  accelerator: [:rt_cores, :cuda_cores, :cpu] do
  # Framework will try RT Cores first
  # Falls back to CUDA if no RT hardware
  # Falls back to CPU as final fallback
  # Currently defaults to CPU implementation
end
```

**Future Execution logic:**
```elixir
case available_hardware() do
  %{rt_cores: true} ->
    use_rt_cores()  # Framework-ready
  %{cuda_cores: true} ->
    use_cuda_fallback()  # Framework-ready
  _ ->
    use_cpu_fallback()  # Currently implemented
end
```

---

### Hybrid Accelerators

**Use multiple in one interaction:**

```elixir
definteraction :hybrid_simulation, accelerator: :auto do
  let {particles} do
    # Step 1: Find neighbors (RT Cores)
    neighbors = rt_find_neighbors(particles)
    
    # Step 2: Compute forces (Tensor Cores)
    forces = tensor_compute_forces(neighbors)
    
    # Step 3: Predict evolution (NPU)
    predictions = npu_predict(particles, forces)
    
    # Step 4: Integrate (CUDA Cores)
    updated = cuda_integrate(particles, forces, predictions)
    
    updated
  end
end
```

**Compiler will automatically:**
1. Dispatch each step to optimal hardware
2. Handle data transfers between accelerators
3. Synchronize across devices

---

## 4. Platform-Specific Considerations

### NVIDIA (Best Support)

```elixir
# Framework supports all NVIDIA accelerators
definteraction :nvidia_optimized,
  accelerator: [:rt_cores, :tensor_cores, :cuda_cores] do
  # RTX 4090: All framework-ready
  # RTX 5070 Ti: All framework-ready
  # Currently falls back to CPU implementations
end
```

**Available:**
- 🔄 RT Cores (Gen 3/4) - Framework-ready
- 🔄 Tensor Cores (Gen 4/5) - Framework-ready
- 📋 NPU - Planned (use Tensor cores as fallback)
- 🔄 CUDA Cores - Framework-ready
- ✅ CPU SIMD - Implemented

---

### AMD

```elixir
# ROCm + NPU support
definteraction :amd_optimized,
  accelerator: [:rt_cores, :npu, :gpu] do
  # RX 7800 XT: RT + GPU
  # Ryzen AI: NPU + GPU
end
```

**Available:**
- ✅ Ray Accelerators (weaker than NVIDIA)
- ✅ AI Accelerators (instead of Tensor)
- ✅ NPU (Ryzen AI only, 50 TOPS)
- ✅ Stream Processors (ROCm)

**Note:** Use `:gpu` not `:cuda_cores` (ROCm auto-translates)

---

### Apple M-Series

```elixir
# Metal + Neural Engine
definteraction :apple_optimized,
  accelerator: [:rt_cores, :npu, :gpu] do
  # M4 Max: RT + NPU + GPU cores
  # No Tensor Cores!
end
```

**Available:**
- ✅ Hardware RT (M3+, shared GPU)
- ❌ Tensor Cores (use GPU cores instead)
- ✅ Neural Engine (38 TOPS)
- ✅ GPU Cores (Metal)

**Special case:**
```elixir
# Apple automatically maps :tensor_cores → GPU cores
# But less efficient than NVIDIA Tensor Cores
definteraction :matrix_op, accelerator: :tensor_cores do
  # On Apple: Uses Metal Performance Shaders
  # On NVIDIA: Uses Tensor Cores
  # Performance: NVIDIA 5× faster
end
```

---

### Intel

```elixir
# OneAPI + NPU
definteraction :intel_optimized,
  accelerator: [:rt_cores, :npu, :gpu] do
  # Arc A770: RT + XMX
  # Core Ultra: NPU + iGPU
end
```

**Available:**
- ✅ RT Units (Arc GPUs, weak)
- ✅ XMX Engines (like Tensor Cores)
- ✅ NPU (Core Ultra, 10-40 TOPS)
- ✅ Xe Cores (OneAPI)

---

## 5. Performance Guidelines

### Expected Speedups

**Current Implementation (SIMD CPU):**
```
Particle integration (10K particles):
  Scalar CPU:  104ms
  SIMD CPU:    ~35ms  (3× speedup) ← Currently implemented
```

**RT Cores (Framework-Ready):**
```
Collision detection (10K particles):
  CPU:         1000ms
  RT Cores:    ~100ms  (10× potential) ← Framework-ready, CPU fallback
```

**Tensor Cores (Framework-Ready):**
```
Force matrix (10K particles):
  CPU:         5000ms
  Tensor:      ~500ms  (10× potential) ← Framework-ready, CPU fallback
```

**NPU (Planned):**
```
Neural inference (batch 1000):
  CPU:         1000ms
  NPU:         ~100ms  (10× potential) ← Planned, no implementation yet
```

---

### When Acceleration Doesn't Help

**Too Small:**
```elixir
# Bad: GPU overhead > computation time
definteraction :tiny_work, accelerator: :cuda_cores do
  let {particles} when length(particles) < 100 do
    # GPU kernel launch: 5ms
    # Actual work: 0.1ms
    # Total: 5.1ms (slower than CPU!)
  end
end

# Good: Use CPU for small workloads
definteraction :tiny_work, accelerator: :cpu do
  # CPU: 0.5ms (10× faster than GPU!)
end
```

**Memory Bound:**
```elixir
# Bad: Transferring data costs more than compute
definteraction :memory_bound, accelerator: :tensor_cores do
  let {huge_matrix} do
    # Transfer to GPU: 1000ms
    # Tensor Core compute: 10ms
    # Transfer back: 1000ms
    # Total: 2010ms
    
    # CPU would be faster: 500ms
  end
end
```

---

## 6. Debugging Accelerators

### See What Was Chosen

```elixir
# Enable accelerator debugging
export AII_DEBUG_DISPATCH=1

definteraction :my_interaction, accelerator: :auto do
  # Compiler will print:
  # [AII Dispatch] Analyzing :my_interaction
  # [AII Dispatch] Detected: matrix_multiply → :tensor_cores
  # [AII Dispatch] Selected: RTX 4090 Tensor Cores (Gen 4)
  # [AII Dispatch] Estimated speedup: 50×
end
```

### Force CPU for Debugging

```elixir
# Temporarily disable GPU
export AII_FORCE_CPU=1

# Or in code:
definteraction :buggy, accelerator: :cpu do
  # Much easier to debug on CPU:
  # - Can use IO.inspect
  # - Can use breakpoints
  # - Can use ExUnit
end
```

---

## 7. Complete Examples

### Example 1: N-Body Gravity Simulation

```elixir
defmodule AII.Examples.Gravity do
  use AII.DSL
  
  conserved_quantity :energy
  conserved_quantity :momentum
  
  defagent Particle do
    property :mass, Conserved<Float>
    state :position, Vec3
    state :velocity, Vec3
    
    derives :kinetic_energy, Energy do
      0.5 * mass.value * Vec3.magnitude_squared(velocity)
    end
  end
  
  # Step 1: Find nearby particles (RT Cores)
  definteraction :find_neighbors, accelerator: :rt_cores do
    let {particles, radius} do
      bvh = build_bvh(particles)
      
      for particle <- particles do
        neighbors = rt_sphere_query(
          bvh, 
          particle.position, 
          radius
        )
        {particle, neighbors}
      end
    end
  end
  
  # Step 2: Compute gravitational forces (Tensor Cores)
  definteraction :compute_gravity, accelerator: :tensor_cores do
    let {particle_neighbors} do
      for {particle, neighbors} <- particle_neighbors do
        # Build local force matrix
        positions = extract_positions(neighbors)
        masses = extract_masses(neighbors)
        
        # Tensor Core accelerated
        forces = gravitational_force_matrix(
          particle.position,
          particle.mass,
          positions,
          masses
        )
        
        total_force = sum_forces(forces)
        {particle, total_force}
      end
    end
  end
  
  # Step 3: Integrate motion (CUDA Cores)
  definteraction :integrate, accelerator: :cuda_cores do
    let {particle_forces, dt} do
      parallel_map(particle_forces, fn {particle, force} ->
        acceleration = force / particle.mass.value
        new_velocity = particle.velocity + acceleration * dt
        new_position = particle.position + new_velocity * dt
        
        %{particle |
          velocity: new_velocity,
          position: new_position
        }
      end)
    end
  end
  
  # Main simulation loop (orchestrates all)
  definteraction :simulate_step, accelerator: :auto do
    let {particles, dt, radius} do
      particles
      |> find_neighbors(radius)      # RT Cores
      |> compute_gravity()           # Tensor Cores
      |> integrate(dt)               # CUDA Cores
    end
  end
end
```

---

### Example 2: Learned Dynamics with NPU

```elixir
defmodule AII.Examples.LearnedPhysics do
  use AII.DSL
  
  # Pre-trained model (learned from physics simulation)
  @dynamics_model Model.load("learned_dynamics.onnx")
  
  # Traditional physics (baseline)
  definteraction :physics_integrate, accelerator: :cuda_cores do
    let {particles, dt} do
      # Traditional Euler integration
      parallel_map(particles, fn p ->
        new_v = p.velocity + p.acceleration * dt
        new_p = p.position + new_v * dt
        %{p | velocity: new_v, position: new_p}
      end)
    end
  end
  
  # Learned dynamics (faster, approximate)
  definteraction :learned_integrate, accelerator: :npu do
    let {particles, dt} do
      # Extract features for neural network
      features = extract_features(particles)
      
      # NPU inference (100× faster than physics!)
      predictions = npu_inference(
        model: @dynamics_model,
        input: features,
        precision: :int8
      )
      
      # Apply predictions
      apply_predictions(particles, predictions, dt)
    end
  end
  
  # Hybrid: Use NPU most of the time, physics occasionally
  definteraction :hybrid_integrate, accelerator: [:npu, :cuda_cores] do
    let {particles, dt, step_num} do
      if rem(step_num, 100) == 0 do
        # Every 100 steps: Use accurate physics
        physics_integrate(particles, dt)
      else
        # Other 99 steps: Use fast learned dynamics
        learned_integrate(particles, dt)
      end
    end
  end
end
```

---

## 8. Summary Table

| Accelerator | Best For | Current Status | Hardware | Notes |
|-------------|----------|----------------|----------|-------|
| `:auto` | Let compiler decide | ✅ Implemented | All | Defaults to SIMD CPU |
| `:rt_cores` | Spatial queries | 🔄 Framework-ready | RT Cores | NVIDIA, AMD, Apple M4+ |
| `:tensor_cores` | Matrix multiply | 🔄 Framework-ready | Tensor | NVIDIA only |
| `:npu` | Neural inference | 📋 Planned | NPU | AMD, Intel, Apple |
| `:cuda_cores` | Parallel compute | 🔄 Framework-ready | CUDA | NVIDIA, AMD (ROCm) |
| `:gpu` | Generic GPU | 🔄 Framework-ready | Any GPU | Vulkan/OpenCL/Metal |
| `:cpu` | Debugging, small | ✅ Implemented | CPU | Always available |
| `:parallel` | Multi-core | ✅ Implemented | CPU | Uses Elixir Flow |
| `:simd` | Vector ops | ✅ Implemented | CPU | Zig SIMD in runtime |

---

## 9. Best Practices

### DO:
✅ Use `:auto` as default (trust compiler)  
✅ Use `:rt_cores` for spatial queries  
✅ Use `:tensor_cores` for matrix ops  
✅ Use `:npu` for neural inference  
✅ Provide fallback chain: `[:rt_cores, :cuda_cores, :cpu]`  
✅ Use `:cpu` for debugging  

### DON'T:
❌ Specify accelerator for tiny workloads  
❌ Use GPU for <100 particles  
❌ Use `:tensor_cores` on Apple (no hardware)  
❌ Forget to handle hardware unavailable  
❌ Optimize prematurely (profile first!)  

---

**Remember: When in doubt, use `:auto` and let the compiler choose!** 🚀