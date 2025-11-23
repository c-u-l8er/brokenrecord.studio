# Benchmarking Guide

## Understanding Performance Metrics

This document explains the different performance metrics reported by BrokenRecord Zero benchmarks and why they may seem inconsistent.

## Types of Benchmarks

### 1. Compilation Benchmarks (`dsl_bench.exs`)

These measure how fast the **compiler** can:
- Parse DSL definitions
- Generate intermediate representation (IR)
- Apply optimization passes
- Generate native C code

**Results**: ~26M operations/sec (DSL parsing and compilation)

### 2. Raw Operation Benchmarks (`test_physics.c`)

This standalone C benchmark measures raw floating-point operations:
- Counts each particle update as one operation
- 10,000 particles × 1000 steps = 10,000,000 operations
- Reports "2.5 billion operations/sec" for simple gravity

**Why so high?**
- Very simple O(N) gravity calculation
- No collision detection (O(N²) complexity)
- Optimized C loops with AVX SIMD
- Measures raw FLOPs, not realistic physics performance

### 3. Simulation Benchmarks (Not Yet Implemented)

These would measure actual physics simulation performance:
- Particle updates per second with full physics
- Collision detection performance
- Memory bandwidth utilization
- Cache miss rates

## Performance Discrepancy Explained

The apparent discrepancy between "2.5 billion ops/sec" (C) and "10M ops/sec" (Elixir) is because:

1. **Different Metrics Being Measured**
   - C benchmark: Raw floating-point operations in optimized loops
   - Elixir benchmark: DSL parsing and compilation speed

2. **Different Complexity**
   - C benchmark: Simple O(N) gravity with basic Euler
   - Real physics: O(N²) collisions, complex forces, constraints

3. **Different Overheads**
   - C benchmark: Zero abstraction, direct memory access
   - Elixir: Full compiler pipeline with type checking

## Realistic Performance Expectations

For a complete physics system with:
- Particle-particle collisions (O(N²) with spatial hashing)
- Multiple force types
- Constraint solving
- Realistic memory layouts

Expected performance: **1-10M particles/sec** on a single CPU core.

## How to Add Real Simulation Benchmarks

```elixir
defmodule BenchmarkTest do
  use ExUnit.Case
  
  test "physics simulation benchmark" do
    # Create a realistic system
    particles = for i <- 1..10000 do
      %{
        position: {:rand.uniform(200) - 100, :rand.uniform(200) - 100, :rand.uniform(200) - 100},
        velocity: {:rand.uniform(2) - 1, :rand.uniform(2) - 1, :rand.uniform(2) - 1},
        mass: 1.0 + :rand.uniform() * 10
      }
    end
    
    initial_state = %{particles: particles}
    
    # Benchmark actual simulation
    {time, result} = :timer.tc(fn ->
      MyPhysics.simulate(initial_state, dt: 0.01, steps: 100)
    end)
    
    particles_per_sec = 10000 * 100 / (time / 1000)
    
    IO.puts("Simulation: #{particles_per_sec} particles/sec")
    
    # Verify conservation
    final_energy = calculate_energy(result.particles)
    initial_energy = calculate_energy(particles)
    
    assert_in_delta(final_energy, initial_energy, 0.001)
  end
  
  defp calculate_energy(particles) do
    # Simple kinetic energy calculation
    Enum.sum(particles, fn p ->
      {vx, vy, vz} = p.velocity
      0.5 * p.mass * (vx*vx + vy*vy + vz*vz)
    end)
  end
end
```

## Optimization Targets

| Component | Current Status | Target | Notes |
|-----------|---------------|--------|-------|
| DSL Compilation | 26.24 M ops/sec | 20+ M ops/sec | ✅ Exceeding target |
| Code Generation | 1.89 M ops/sec | 5+ M ops/sec | ✅ Good |
| Native Compilation | Not measured | 100+ M ops/sec | 🚧 Needs work |
| Full Simulation | Not measured | 10+ M particles/sec | 🚧 Needs work |

## Profiling Recommendations

1. **Use `perf` on Linux**:
   ```bash
   perf stat -e cycles,instructions,cache-misses ./my_app
   ```

2. **Use Instruments on macOS**:
   ```bash
   xctrace record --launch ./my_app
   ```

3. **Profile Memory Usage**:
   ```elixir
   :erlang.memory()  # Check before/after
   ```

4. **Benchmark with Realistic Data**:
   - Include collision detection
   - Use varied particle sizes
   - Test with different timesteps
   - Measure memory bandwidth