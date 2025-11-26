# BrokenRecord Zero - Development Guide

## Current Status

✅ **Working Components:**
- DSL parser and compiler
- IR generation and optimization
- Native code generation with SIMD
- NIF integration
- End-to-end tests
- Benchmarks

  - **Actor Model & DSL Benchmarks**: DSL compilation ~30M ops/sec, large actor system (1000 actors) ~4.7k ips. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)
## Architecture Overview

BrokenRecord Zero is a compile-time physics optimizer for Elixir. It transforms high-level physics descriptions into optimized native code.

### Core Components

1. **DSL Layer** (`lib/broken_record/zero/dsl.ex`)
   - Defines physics systems with agents and rules
   - Supports type annotations and conservation declarations

2. **IR Layer** (`lib/broken_record/zero/ir.ex`)
   - Intermediate representation for optimization
   - Supports type checking and analysis

3. **Optimizer** (`lib/broken_record/zero/optimizer.ex`)
   - Applies optimization passes (SIMD, spatial hashing, etc.)
   - Computes optimal memory layouts

4. **Code Generator** (`lib/broken_record/zero/code_gen.ex`)
   - Generates optimized C code from IR
   - Emits SIMD intrinsics and cache-friendly structures

5. **Runtime** (`lib/broken_record/zero/runtime.ex`)
   - Bridges Elixir and native code via NIF
   - Handles data conversion between formats

## Development Workflow

### Adding New Features

1. **DSL Extensions**
   - Add new constructs in `dsl.ex`
   - Update parser for new syntax

2. **Optimization Passes**
   - Implement in `optimizer.ex`
   - Register in `optimize/2` function

3. **Target Backends**
   - Add code generator for new architecture
   - Update compilation pipeline

### Testing

```bash
# Run all tests
mix test

# Run specific test
mix test test/broken_record/e2e_test.exs

# Run benchmarks
mix run benchmarks/dsl_bench.exs
```

### Debugging

1. **Compilation Issues**
   - Check IR generation in `ir.ex`
   - Verify optimization passes
   - Examine generated C code in `/tmp/`

2. **Runtime Issues**
   - Check NIF loading in `runtime.ex`
   - Verify data conversion functions

3. **Performance Issues**
   - Profile with `perf` on Linux
   - Check SIMD code generation

## Performance Optimization

### Current Optimizations

1. **SIMD Vectorization**
   - AVX instruction set for 8-float parallel processing
   - Automatic vector width detection

2. **Memory Layout**
   - Structure of Arrays (SoA) for cache efficiency
   - 64-byte alignment for AVX

3. **Parallelization**
   - OpenMP support for multi-core CPUs
   - Work-stealing scheduler design

### Future Optimizations

1. **Spatial Hashing**
   - O(N) collision detection
   - Grid-based or octree implementations

2. **GPU Support**
   - CUDA backend for massive parallelism
   - Thrust for GPU memory management

## Code Organization

```
lib/broken_record/zero/
├── dsl.ex          # High-level DSL definitions
├── ir.ex            # Intermediate representation
├── analyzer.ex       # Type checking and conservation
├── optimizer.ex      # Optimization passes
├── code_gen.ex       # C code generation
└── runtime.ex        # NIF interface and execution

examples/
├── particle_system.ex    # Basic particle simulation
├── collision_simulation.ex # Collision detection demo
└── gravity_simulation.ex   # Gravity field demo

benchmarks/
├── dsl_bench.exs         # Full compiler benchmarks
├── dsl_bench_simple.exs # Simple performance test
└── broken_record_bench.exs  # Native code benchmarks

test/
├── e2e_test.exs          # End-to-end system tests
├── zero_test.exs          # Unit tests for components
├── compiler_test.exs       # Compilation pipeline tests
└── performance_test.exs    # Performance regression tests
```

## Build System

### Native Compilation

The system generates optimized C code and compiles it to a shared library:

```bash
# Generated code includes:
- SIMD kernels with AVX intrinsics
- Cache-friendly SoA memory layout
- OpenMP parallelization directives
- NIF interface for Elixir integration

gcc -O3 -march=native -ffast-math -fopenmp \
  -shared -fPIC -o priv/native.so /tmp/broken_record_native.c
```

### Elixir Integration

```elixir
# NIF loads automatically when module is used
defmodule MyPhysics do
  use BrokenRecord.Zero
  
  # Physics system defined here
end

# NIF provides native implementations
MyPhysics.simulate(particles, dt: 0.01, steps: 1000)
```

## Performance Guidelines

### Achieving Target Performance

1. **Batch Operations**
   - Add all particles before simulation
   - Minimize state changes

2. **Optimize Timestep**
   - Larger dt = fewer steps = less overhead
   - Balance accuracy vs performance

3. **Reuse Systems**
   - Create once, simulate many times
   - Avoid repeated compilation

### Profiling

```bash
# Linux perf
perf stat -e cycles,instructions,cache-misses ./my_app

# macOS Instruments
xctrace record --launch ./my_app
```

## Troubleshooting

### Common Issues

1. **NIF Loading Errors**
   ```bash
   # Check if native.so exists
   ls priv/native.so
   
   # Check NIF symbols
   nm -D priv/native.so | grep nif_init
   ```

2. **Compilation Failures**
   ```bash
   # Check GCC version
   gcc --version  # Need 4.8+ for full AVX
   
   # Check AVX support
   cat /proc/cpuinfo | grep avx
   ```

3. **Performance Issues**
   - Verify SIMD code generation
   - Check memory alignment
   - Profile optimization passes

## Release Process

1. Update version in `mix.exs`
2. Update CHANGELOG.md
3. Tag release: `git tag v1.0.0`
4. Publish to Hex: `mix hex.publish`

## Architecture Decisions

### Why Structure of Arrays?

- **Cache Efficiency**: Each array is a cache line
- **SIMD Alignment**: Natural 256/512-bit boundaries
- **Vectorization**: Same operation on all elements

### Why NIF Instead of Ports?

- **Performance**: Zero-copy data sharing
- **Simplicity**: Direct function calls
- **Flexibility**: Easy state management

### Why Compile-Time Optimization?

- **Zero Overhead**: No runtime compilation
- **Global View**: Optimizer sees entire system
- **Verification**: Prove properties before execution