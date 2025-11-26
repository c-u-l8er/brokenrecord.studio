# BrokenRecord Zero API Reference

High-performance physics engine for Elixir.

## Performance Summary

Updated benchmarks:

**Actor Model Runtime:**
| System | Avg Time (ms) |
|--------|---------------|
| 1000 actors, 100 steps | 211 |
| 10k messages | 1,160 |

**DSL Compilation:**
| Stage | Avg (μs) |
|-------|----------|
| Simple Compile | 33 |
| Code Gen | 546 |

[Full Benchmarks](../benchmarks/)

## Core DSL

### defsystem

```elixir
defsystem MySystem do
  compile_target :cpu
  optimize [:simd]

  agents do
    defagent Particle do
      field :position, :vec3
    end
  end

  rules do
    interaction update(p: Particle, dt: float) do
      p.position = p.position + {0, -9.81 * dt, 0}
    end
  end
end
```

## Runtime API

- `MySystem.simulate(particles, dt: 0.01, steps: 1000)`
- `MySystem.create_particles(n: 1000)`

## Compilation Targets

- `:cpu` - SIMD optimized
- `:cuda` - GPU (future)

## Optimization Passes

- `:simd` - Vectorization
- `:spatial_hash` - Collision opt

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for details.

[Full docs](../README.md)