# AII (Artificial Interaction Intelligence)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/c-u-l8er/brokenrecord.studio/ci.yml)](https://github.com/c-u-l8er/brokenrecord.studio/actions)
[![Hex.pm Version](https://img.shields.io/hexpm/v/aii.svg)](https://hex.pm/packages/aii)

**High-level physics DSL for Elixir.** Defines agents, interactions, and conservation laws with compile-time verification and hardware-accelerated runtime simulation.

[🌐 Live Demo](index.html) | [📚 Examples](examples.html) | [📖 Docs](scrap/docs/design.md)

## ✨ Features

- ✅ **Conservation Guaranteed**: Energy/momentum verified at compile time
- 🎯 **Type-Safe DSL**: High-level physics definitions with compile-time guarantees
- 🔧 **Hardware Dispatch**: Automatic selection of CPU/GPU/CUDA/RT cores
- 🧬 **Agent-Based Modeling**: Define particles, interactions, and physical laws
- ⚛️ **Chemical Reactions**: Mass-conserving reaction networks
- 🚀 **NIF Runtime**: High-performance Zig-based particle physics engine
- 📊 **Performance Monitoring**: Real-time hardware utilization and conservation verification

## 🚀 Quick Start

### 1. Add to `mix.exs`

```elixir
defp deps do
  [
    {:aii, "~> 0.1.0"}
  ]
end
```

### 2. Basic Physics Simulation

```elixir
defmodule MyPhysics do
  use AII

  defagent Particle do
    field :position, Vec3
    field :velocity, Vec3
    field :mass, Energy
    conserves [:energy, :momentum]
  end

  definteraction gravity(p1 :: Particle, p2 :: Particle) do
    r_vec = p2.position - p1.position
    r = magnitude(r_vec)
    force = 6.67e-11 * p1.mass * p2.mass / (r * r)
    dir = normalize(r_vec)

    p1.velocity = p1.velocity + dir * force * dt / p1.mass
    p2.velocity = p2.velocity - dir * force * dt / p2.mass
  end

  definteraction integrate(p :: Particle) do
    p.position = p.position + p.velocity * dt
  end
end

# Run simulation
particles = [
  %{position: {0.0, 0.0, 10.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0, energy: 0.5, id: 1},
  %{position: {5.0, 0.0, 10.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0, energy: 0.5, id: 2}
]

{:ok, result} = AII.run_simulation(MyPhysics, steps: 1000, dt: 0.01, particles: particles)
IO.inspect(result)
```

## 📊 Performance Benchmarks

Performance benchmarks will be available once the full Zig runtime implementation is complete and tested.

See [benchmarks/](benchmarks/) for future performance reports.

## 🏗️ Architecture

```
High-Level Elixir DSL
         ↓ (compile-time)
Type Checking & Conservation Verification
         ↓ (runtime)
Zig NIF Particle Physics Engine
         ↓ (hardware dispatch)
CPU/GPU/CUDA/RT Core Acceleration
```

## 🔧 Build & Test

### Prerequisites

- Elixir 1.14+
- Erlang 25+
- Zig 0.11+ (for NIF compilation)

### Setup

```bash
mix deps.get
mix compile          # Builds Zig NIF
mix test             # Runs 184 comprehensive tests
```

### Development

```bash
# Run specific test suite
mix test test/aii/

# Run examples
mix run examples/my_physics.ex

# Build NIF manually
cd runtime/zig && zig build
```

## 📚 Examples

- [Particle Physics](examples/aii_particle_physics.ex) - Basic particle interactions
- [Chemical Reactions](examples/aii_chemical_reactions.ex) - Mass-conserving networks
- [Hardware Dispatch](examples/aii_hardware_dispatch.ex) - Multi-accelerator simulation
- [Conservation Verification](examples/aii_conservation_verification.ex) - Physics law checking

[View Interactive Examples](examples.html)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

**Priority Areas:**
- Complete Zig runtime implementation
- GPU/CUDA acceleration
- Performance benchmarking
- Visualization and monitoring tools

## 📄 License

MIT © brokenrecord.studio

## Quick Commands

```sh
# Compile the Zig NIF
cd runtime/zig && zig build

# Copy to priv directory
cp runtime/zig/zig-out/lib/libaii_runtime.so priv/
```