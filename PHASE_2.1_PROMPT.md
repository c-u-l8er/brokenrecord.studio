# Phase 2.1: Complete Zig NIF Integration & Runtime Implementation

## Overview

Phase 2 focused on implementing the Zig runtime for particle physics simulation, but the integration is incomplete. The Elixir NIF module (`lib/aii/nif.ex`) contains mocked functions, and the Zig NIF (`runtime/zig/nif.zig`) has stub implementations for `add_particle` and `get_particles`. This phase completes the Zig NIF integration, ensuring real physics computation with conservation verification.

## Current Status Assessment

### ✅ Completed from Phase 2
- **Zig Particle System**: Core `ParticleSystem` struct with Euler integration, energy conservation checks, and force application (`runtime/zig/particle_system.zig`)
- **NIF Skeleton**: Basic NIF functions for `create_particle_system`, `integrate`, and `destroy_system` are implemented in Zig
- **Elixir Runtime**: `AII.run_simulation/2` calls NIF functions, but falls back to mocks

### ❌ Incomplete/Incorrect
- **NIF Loading**: `lib/aii/nif.ex` has `@on_load :load_nif` returning `:ok`, but doesn't load the actual Zig NIF (should use Zigler or manual loading)
- **add_particle**: Zig implementation is a stub; doesn't parse Elixir map into `Particle` struct
- **get_particles**: Zig implementation returns empty list; doesn't convert `Particle` structs to Elixir terms
- **Beam Integration**: `beam.zig` may have issues; needs verification for term parsing
- **Conservation**: Runtime conservation checks are in place but untested with real data

### 📋 Additional Issues
- **Zigler Usage**: Project uses Zigler dependency but NIF module doesn't use Zigler macros (`use Zig`)
- **Build Integration**: Ensure `mix compile` builds and loads the Zig NIF correctly
- **Data Conversion**: Implement robust conversion between Elixir maps and Zig structs

## Phase 2.1 Goals

### Primary Objectives
1. **Complete NIF Functions**: Implement `add_particle` and `get_particles` in Zig with proper Elixir term handling
2. **Fix NIF Loading**: Update `lib/aii/nif.ex` to properly load the Zig NIF using Zigler or manual loading
3. **Test Real Simulation**: Ensure `AII.run_simulation/2` performs actual particle physics with conservation verification
4. **Validate Conservation**: Test that energy/momentum conservation holds in real simulations

### Success Criteria
- `AII.NIF.add_particle/2` and `AII.NIF.get_particles/1` work without `:not_loaded` errors
- `AII.run_simulation/2` returns actual particle state changes from Zig computation
- Conservation violations are detected and reported correctly
- All 184 tests pass with real runtime (no mocks)
- Benchmarks show measurable physics ops/sec

## Detailed Development Tasks

### Phase 2.1.1: Fix NIF Loading in Elixir
- **File**: `lib/aii/nif.ex`
- **Action**: Replace mocked `load_nif` with proper Zigler integration
- **Requirements**: Use `use Zig` and define Zig functions directly, or manually load the shared library
- **Verification**: `mix compile` loads the NIF successfully, functions are callable

### Phase 2.1.2: Implement add_particle in Zig
- **File**: `runtime/zig/nif.zig`
- **Action**: Parse Elixir map (`argv[1]`) into `Particle` struct and add to system
- **Requirements**: Extract `:position`, `:velocity`, `:mass`, `:energy`, `:id` from Elixir map using `beam.zig` helpers
- **Verification**: Adding particles via NIF updates system state correctly

### Phase 2.1.3: Implement get_particles in Zig
- **File**: `runtime/zig/nif.zig`
- **Action**: Convert `Particle` structs to Elixir list of maps
- **Requirements**: Create Elixir terms for position/velocity tuples, mass/energy floats, id integers
- **Verification**: Retrieved particles match added data

### Phase 2.1.4: Update Beam.zig for Robust Term Handling
- **File**: `runtime/zig/beam.zig`
- **Action**: Add helpers for parsing maps and creating complex terms
- **Requirements**: Functions to get map values, create tuples/lists from Vec3 structs
- **Verification**: Term conversion works for all particle fields

### Phase 2.1.5: Test Real Simulation Loop
- **Action**: Run `AII.run_simulation/2` with simple test case and verify particle movement
- **Requirements**: Positions/velocities change realistically, conservation holds
- **Verification**: No `:not_loaded` errors, actual computation occurs

### Phase 2.1.6: Update Benchmarks
- **File**: `benchmarks/` (update existing)
- **Action**: Replace mock benchmarks with real physics measurements
- **Requirements**: Measure actual ops/sec for different particle counts and interactions
- **Verification**: Performance reports show real throughput

## Development Workflow

### Recommended Order
1. Fix NIF loading (2.1.1) - Critical for any NIF calls
2. Implement add_particle (2.1.2) - Needed to populate systems
3. Implement get_particles (2.1.3) - Needed to return results
4. Enhance beam.zig (2.1.4) - Support for complex data
5. Test simulation (2.1.5) - End-to-end validation
6. Update benchmarks (2.1.6) - Performance measurement

### Testing Strategy
- **Unit Tests**: Test individual NIF functions with real Zig code
- **Integration Tests**: Full simulation runs with conservation checks
- **Performance Tests**: Benchmark real physics vs. mock placeholders
- **Conservation Tests**: Verify laws hold under various conditions

### Dependencies
- **Zigler**: Ensure proper usage for NIF compilation and loading
- **Erlang/Elixir**: Compatible versions for NIF interface

## Risk Assessment

### High Risk
- **Term Parsing**: Complex Elixir ↔ Zig data conversion, potential crashes
- **Memory Management**: Zig allocator interactions with Erlang GC

### Medium Risk  
- **Conservation Accuracy**: Floating-point precision in energy calculations
- **Performance**: Initial Zig implementation may be slower than optimized C++

### Low Risk
- **NIF Loading**: Standard Zigler patterns, well-documented

## Success Metrics

- ✅ `mix test` passes 100% with real NIF implementations
- ✅ `AII.run_simulation/2` computes actual physics (positions/velocities change)
- ✅ Conservation violations detected in test cases
- ✅ Benchmarks show >1M particles/sec for simple simulations
- ✅ No `:not_loaded` or mock fallbacks in production code

This phase finalizes the Zig runtime integration, transforming AII from a DSL framework with mocked runtime into a fully functional high-performance physics simulation engine.