# AII Migration: BrokenRecord C → Zig/Elixir
## Document 1: Migration Overview

### Current System (v1)
```
Elixir DSL → Compiler → IR → C Runtime (NIF) → Execution
├─ dsl.ex: defagent, interaction macros
├─ compiler.ex: DSL → IR transformation
├─ ir.ex: Intermediate representation
├─ codegen.ex: IR → C code generation
└─ brokenrecord_physics.c: Runtime (particles, forces, integration)
```

### Target System (v2 - AII)
```
Elixir AII DSL → Compiler → SPIR-V + Zig → Multi-Device Execution
├─ aii_dsl.ex: Particle-based DSL with conservation types
├─ conservation_checker.ex: Compile-time conservation verification
├─ hardware_dispatcher.ex: Automatic RT/Tensor/NPU/CUDA dispatch
├─ spirv_codegen.ex: Generate GPU shaders
└─ zig_runtime/: Memory-safe native runtime
    ├─ particle_system.zig: Core particle engine
    ├─ vulkan_backend.zig: RT Cores, Tensor Cores
    ├─ npu_backend.zig: Neural inference
    └─ conservation.zig: Runtime verification
```

---

## Critical Changes

### 1. **DSL Transformation: Tokens → Particles**

**OLD (C-based):**
```elixir
defagent Particle do
  field :position, :vec3
  field :velocity, :vec3
end

interaction gravity(p: Particle, dt: float) do
  p.velocity = p.velocity + {0.0, -9.81 * dt, 0.0}
end
```

**NEW (AII):**
```elixir
defagent Particle do
  property :mass, Float, invariant: true
  state :position, Vec3
  state :velocity, Vec3
  state :information, Conserved<Float>  # NEW: Conservation type
  
  derives :energy, Energy do
    0.5 * mass * magnitude(velocity) ** 2
  end
  
  conserves :energy, :momentum, :information  # NEW: Explicit conservation
end

definteraction :gravity, accelerator: :auto do  # NEW: Hardware hint (auto is recommended)
  # Compiler verifies conservation automatically
end
```

**Key Differences:**
- Add `Conserved<T>` type wrapper
- Add `property` (invariant) vs `state` (mutable)
- Add `derives` for computed quantities
- Add `accelerator:` hints for hardware dispatch

---

### 2. **Runtime: C → Zig**

**Why Zig?**
- Memory safety without garbage collection
- C interop (can keep existing C during transition)
- Better error handling (no undefined behavior)
- Compile-time guarantees

**OLD (brokenrecord_physics.c):**
```c
typedef struct {
    float position[3];
    float velocity[3];
    float mass;
} Particle;

void integrate_euler(Particle* particles, int count, float dt) {
    for (int i = 0; i < count; i++) {
        particles[i].position[0] += particles[i].velocity[0] * dt;
        particles[i].position[1] += particles[i].velocity[1] * dt;
        particles[i].position[2] += particles[i].velocity[2] * dt;
    }
}
```

**NEW (particle_system.zig):**
```zig
const Particle = struct {
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    energy: Conserved(f32),  // Conservation type
};

pub fn integrateEuler(
    particles: []Particle,
    dt: f32,
    allocator: Allocator
) !void {
    // Track conservation
    const total_energy_before = computeTotalEnergy(particles);
    
    for (particles) |*p| {
        p.position = p.position.add(p.velocity.mul(dt));
    }
    
    // Verify conservation
    const total_energy_after = computeTotalEnergy(particles);
    if (!conserved(total_energy_before, total_energy_after)) {
        return error.ConservationViolation;
    }
}
```

---

### 3. **Hardware Dispatch: New Layer**

**NEW Component:** `hardware_dispatcher.ex`

```elixir
defmodule AII.HardwareDispatcher do
  @doc "Analyze interaction and choose optimal hardware"
  def dispatch(interaction) do
    cond do
      spatial_query?(interaction) -> :rt_cores
      matrix_operation?(interaction) -> :tensor_cores
      learned_dynamics?(interaction) -> :npu
      true -> :cuda_cores
    end
  end
  
  defp spatial_query?(interaction) do
    # Check if interaction involves collision detection,
    # nearest neighbor queries, BVH traversal
    ast_contains?(interaction.body, [:collision, :nearby, :within_radius])
  end
end
```

**Usage in Codegen:**
```elixir
def generate_interaction(interaction) do
  case HardwareDispatcher.dispatch(interaction) do
    :rt_cores -> generate_ray_query(interaction)
    :tensor_cores -> generate_tensor_op(interaction)
    :npu -> generate_npu_inference(interaction)
    :cuda_cores -> generate_cuda_kernel(interaction)
  end
end
```

---

### 4. **Conservation Type System**

**NEW Component:** `conservation_checker.ex`

```elixir
defmodule AII.ConservationChecker do
  @doc "Verify conservation at compile time"
  def verify_interaction(interaction, conserved_quantities) do
    # Track each conserved quantity through AST
    for quantity <- conserved_quantities do
      case track_quantity(interaction.body, quantity) do
        {:ok, :conserved} -> :ok
        {:error, :violation} -> 
          raise CompileError, 
            message: "Conservation violated: #{quantity}"
      end
    end
  end
  
  defp track_quantity(ast, quantity) do
    # Build symbolic representation
    # Verify: total_before == total_after
  end
end
```

---

## Migration Path

### Phase 1: Parallel Development (Month 1)
- Keep C runtime working
- Build Zig runtime alongside
- New DSL macros (Conserved<T>, property, derives)
- Conservation checker (compile-time only)

### Phase 2: Zig Integration (Month 2)
- Port C functions to Zig one-by-one
- Zig NIFs call existing C when needed
- Test parity: Zig results == C results

### Phase 3: Hardware Dispatch (Month 3)
- Add Vulkan backend (RT Cores, Tensor Cores)
- Hardware dispatcher analyzes code
- Generate SPIR-V for GPU

### Phase 4: NPU Support (Month 4)
- Platform-specific NPU bindings
- Learned dynamics models
- Hybrid CPU/GPU/NPU execution

### Phase 5: Full AII (Month 5-6)
- Remove C runtime entirely
- Pure Zig + Elixir
- All hardware accelerators working
- Conservation guaranteed

---

## File Structure Changes

```
lib/
├─ aii/
│  ├─ dsl.ex                    # NEW: Particle-based DSL
│  ├─ conservation_checker.ex   # NEW: Compile-time verification
│  ├─ hardware_dispatcher.ex    # NEW: Hardware selection
│  ├─ compiler.ex               # MODIFIED: Add conservation passes
│  ├─ codegen/
│  │  ├─ spirv.ex              # NEW: GPU shader generation
│  │  ├─ zig.ex                # NEW: Zig code generation
│  │  └─ cuda.ex               # KEEP: CUDA fallback
│  └─ types.ex                 # NEW: Conserved<T>, Energy, etc.
│
runtime/
├─ zig/                        # NEW: Replace C entirely
│  ├─ particle_system.zig
│  ├─ conservation.zig
│  ├─ vulkan_backend.zig
│  ├─ npu_backend.zig
│  └─ build.zig
│
└─ c/                          # LEGACY: Remove after Phase 2
   └─ brokenrecord_physics.c
```

---

## Key Implementation Notes

### 1. Conserved<T> Type
```elixir
# In types.ex
defmodule AII.Types.Conserved do
  @type t(value_type) :: %__MODULE__{
    value: value_type,
    tracked: boolean()
  }
  
  defstruct value: 0, tracked: true
  
  # Compiler enforces: can only transfer, never create/destroy
end
```

### 2. Hardware Accelerator Hints
```elixir
# Decorator syntax with comprehensive options
definteraction :collide, accelerator: :rt_cores do
  # RT Cores for spatial queries
end

definteraction :matrix_ops, accelerator: :tensor_cores do
  # Tensor Cores for matrix operations
end

definteraction :neural_inference, accelerator: :npu do
  # NPU for neural network inference
end

definteraction :general_gpu, accelerator: :gpu do
  # Vendor-agnostic GPU
end

definteraction :multi_core, accelerator: :parallel do
  # Multi-core CPU parallelism
end

definteraction :vector_ops, accelerator: :simd do
  # SIMD vector instructions
end

# Recommended: Let compiler choose
definteraction :auto_dispatch, accelerator: :auto do
  # Compiler analyzes and selects optimal hardware
end

# Fallback chain
definteraction :with_fallback, accelerator: [:rt_cores, :cuda_cores, :cpu] do
  # Tries RT Cores, falls back to CUDA, then CPU
end
```

### 3. SPIR-V Generation
```elixir
# In spirv.ex
def generate_compute_shader(interaction) do
  """
  #version 450
  layout(local_size_x = 256) in;
  
  layout(binding = 0) buffer Particles {
    vec3 positions[];
  };
  
  void main() {
    uint idx = gl_GlobalInvocationID.x;
    // Generated from interaction body
  }
  """
end
```

---

## Success Criteria

**Must Have:**
- [ ] Conservation types work (Conserved<T>)
- [ ] Compile-time conservation verification
- [ ] Zig runtime parity with C
- [ ] GPU execution (Vulkan)
- [ ] RT Cores for collision detection
- [ ] All accelerator types working (:auto, :rt_cores, :tensor_cores, :npu, :cuda_cores, :gpu, :cpu, :parallel, :simd)

**Should Have:**
- [ ] Tensor Cores for matrix ops
- [ ] NPU for inference
- [ ] Automatic hardware dispatch with fallback chains
- [ ] Platform-specific optimizations (NVIDIA, AMD, Apple, Intel)
- [ ] 100× speedup over pure Elixir
- [ ] Hardware capability detection

**Nice to Have:**
- [ ] Multi-GPU support
- [ ] Distributed execution (BEAM)
- [ ] Hot code reload
- [ ] Dynamic hardware switching
- [ ] Performance profiling per accelerator

---

## Critical Path (Next Developer)

1. **Start here:** Implement `AII.Types.Conserved` module
2. **Then:** Add conservation tracking to DSL macros
3. **Then:** Build Zig runtime (particle_system.zig)
4. **Then:** Connect Elixir → Zig via NIFs
5. **Then:** Add Vulkan backend for GPU
6. **Finally:** Hardware dispatcher + SPIR-V codegen
