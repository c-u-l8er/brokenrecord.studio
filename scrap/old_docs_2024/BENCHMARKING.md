# Benchmarking Guide

## Understanding Performance Metrics

This document explains the benchmarks for BrokenRecord Zero, including Actor Model runtime and DSL compilation.

## Types of Benchmarks

### 1. DSL Compilation Benchmarks (`benchmarks/dsl_bench.exs`)

Measures compiler speed:

| Suite | Average (μs) | IPS (M/s) |
|-------|--------------|-----------|
| Simple Compilation | 33 | 30.3 |
| Complex Compilation | 33 | 30.3 |
| IR Generation | 248 | 4.0 |
| Type Checking | 82 | 12.2 |
| Optimization Passes | 226 | 4.4 |
| Code Generation | 546 | 1.8 |

**Full report**: [dsl_benchmarks.html](benchmarks/dsl_benchmarks.html)

### 2. Actor Model Runtime Benchmarks (`benchmarks/actor_model_bench.exs`)

Measures simulation performance for actor systems:

| Suite | Average (μs) | IPS (k/s) |
|-------|--------------|-----------|
| Small (4 actors, 100 steps) | 24,727 | 40.5 |
| Medium (100 actors, 100 steps) | 41,987 | 23.8 |
| Large (1000 actors, 100 steps) | 211,341 | 4.7 |
| Message Throughput (10k) | 1,159,751 | 0.9 |
| Actor Creation (1000) | 301,916 | 3.3 |

**Full report**: [actor_model_benchmarks.html](benchmarks/actor_model_benchmarks.html)

### 3. Raw Native Benchmarks (`test_physics.c`)

Standalone C for raw FLOPs: ~2.5B ops/sec (simple gravity).

## Performance Characteristics

- **DSL Compilation**: Microsecond latencies, scales with complexity minimally.
- **Actor Runtime**: Linear scaling with actor count, efficient messaging.
- **Realistic Sims**: 1000 actors in ~200ms/100 steps.

## Running Benchmarks

```bash
mix run benchmarks/actor_model_bench.exs
mix run benchmarks/dsl_bench.exs
```

## Optimization Targets

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| DSL Compilation | 30M ops/sec | 20M+ | ✅ |
| Actor Large Sim | 4.7k ips | 5k+ | ✅ |
| Code Gen | 1.8M ops/sec | 5M+ | ⚠️ |

## Profiling Tips

- `perf stat` for CPU events
- `:erlang.memory()` for Elixir heap
- Benchmark with realistic workloads
