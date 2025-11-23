# BrokenRecord Zero
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/c-u-l8er/brokenrecord.studio/ci.yml)](https://github.com/c-u-l8er/brokenrecord.studio/actions)
[![Hex.pm Version](https://img.shields.io/hexpm/v/broken_record_zero.svg)](https://hex.pm/packages/broken_record_zero)

**Zero-overhead physics compiler for Elixir.** Transforms high-level physics DSL into optimized native code with compile-time conservation guarantees.

[🌐 Live Demo](index.html) | [📚 Examples](examples.html) | [📊 Benchmarks](benchmarks/) | [📖 Docs](scrap/docs/design.md)

## ✨ Features

- ⚡ **Blazing Fast**: Raw C backend achieves **2.5B operations/sec** (10k particles × 1000 steps)
- ✅ **Conservation Guaranteed**: Energy/momentum verified at **compile time**
- 🎯 **Zero Runtime Overhead**: DSL abstractions erased during compilation
- 🔧 **SIMD Optimized**: AVX2 vectorization processes 8 particles/instruction
- 🧬 **Actor Model**: Full actor system with supervision trees & load balancing
- ⚛️ **Chemical Reactions**: Mass-conserving reaction networks
- 🚀 **GPU Ready**: CUDA target (coming soon)

## 🚀 Quick Start

### 1. Add to `mix.exs`

```elixir
defp deps do
  [
    {:broken_record_zero, "~> 0.1.0"}
  ]
end
```

### 2. Basic Physics Simulation

```elixir
defmodule GravitySim do
  use BrokenRecord.Zero

  defsystem NBody do
    compile_target :cpu
    optimize [:simd, :spatial_hash]

    agents do
      defagent Particle do
        field :position, :vec3
        field :velocity, :vec3
        field :mass, :float
        conserves [:energy, :momentum]
      end
    end

    rules do
      interaction gravity(p1: Particle, p2: Particle, dt: float) do
        r_vec = p2.position - p1.position
        r = length(r_vec)
        force = 6.67e-11 * p1.mass * p2.mass / (r * r)
        dir = r_vec / r
        
        p1.velocity += dir * force * dt / p1.mass
        p2.velocity -= dir * force * dt / p2.mass
      end

      interaction integrate(p: Particle, dt: float) do
        p.position += p.velocity * dt
      end
    end
  end
end

# Run simulation
particles = [
  %{position: {0,0,10}, velocity: {1,0,0}, mass: 1.0},
  %{position: {5,0,10}, velocity: {-1,0,0}, mass: 1.0}
]

result = GravitySim.NBody.simulate(particles, steps: 1000, dt: 0.01)
IO.inspect(result)
```

## 📊 Performance Benchmarks

| Benchmark | Speed | Details |
|-----------|-------|---------|
| **Raw C (10k particles)** | **2.5B ops/sec** | 10k particles × 1000 steps |
| **DSL Compilation (Simple)** | **26M ops/sec** | Parse → IR → Codegen |
| **Actor Model (1000 actors)** | **1.2M steps/sec** | Message passing + supervision |
| **N-Body Gravity** | **500k particles/sec** | Full physics simulation |

See detailed [benchmark reports](benchmarks/).

## 🏗️ Architecture

```
High-Level Elixir DSL
         ↓ (compile-time)
Optimized C (SIMD/AVX2)
         ↓ (NIF)
Zero-overhead Elixir Runtime
```

## 🔧 Build & Test

```bash
mix deps.get
mix compile          # Builds NIF
mix test             # Unit tests
mix run benchmarks/  # Performance
mix phx.server       # Example server
```

## 📚 Examples

- [Actor Model](examples/actor_model.ex) - Concurrent actors with supervision
- [Chemical Reactions](examples/chemical_reaction_net.ex) - Mass-conserving networks
- [N-Body Gravity](examples/gravity_simulation.ex) - Solar system simulation
- [Custom Physics](examples/my_physics.ex) - Collisions + GPU-ready

[View Interactive Examples](examples.html)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

**Priority Areas:**
- GPU (CUDA) backend
- Rigid body dynamics
- Constraint solvers
- Visualization tooling

## 📄 License

MIT © brokenrecord.studio