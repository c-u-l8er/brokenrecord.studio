# BrokenRecord.Zero Examples

This directory contains working examples of the BrokenRecord.Zero physics programming language.

## How to Run Examples

### Option 1: Interactive Web Interface (Recommended)

1. Open `examples.html` in your web browser
2. Click on any example in the sidebar to view the code
3. Click the "🚀 Run Example" button to see simulated console output
4. The console output shows what each example would produce when executed

### Option 2: Direct Elixir Execution

1. Start an Elixir session with `iex -S mix`
2. Load and run examples directly:

```elixir
# Actor Model
Examples.ActorModel.actor_system() |> Examples.ActorModel.simulate(steps: 100, dt: 0.01)

# Chemical Reaction Network  
Examples.ChemicalReactionNet.chemical_mixture() |> Examples.ChemicalReactionNet.simulate(steps: 1000, dt: 0.01)

# Gravity Simulation
Examples.GravitySimulation.solar_system() |> Examples.GravitySimulation.simulate(steps: 1000, dt: 0.01)

# Custom Physics
# See the Demo module in my_physics.ex for usage
```

## Available Examples

### 1. Actor Model (`actor_model.ex`)
- **Concept**: Actor-based concurrency using Lafont interaction nets
- **Features**: Message passing, supervision trees, fault tolerance, load balancing
- **Performance**: Large system (1000 actors, 100 steps): avg 211ms; 10k messages: avg 1.16ms ([benchmarks](benchmarks/actor_model_benchmarks.html))
- **Output**: Actor states, message counts, system statistics

### 2. Chemical Reaction Network (`chemical_reaction_net.ex`)
- **Concept**: Chemical reaction simulation with conservation laws
- **Features**: Synthesis, decomposition, catalysis, thermal dynamics
- **Performance**: DSL compilation <50μs ([benchmarks](benchmarks/dsl_benchmarks.html))
- **Output**: Molecule counts, mass conservation

### 3. Gravity Simulation (`gravity_simulation.ex`)
- **Concept**: N-body gravity with conservation guarantees
- **Features**: Gravitational forces, energy/momentum conservation
- **Performance**: Leverages fast DSL pipeline
- **Output**: Energy levels, conservation verification

### 4. Custom Physics System (`my_physics.ex`)
- **Concept**: Complete physics with GPU support
- **Features**: Collisions, wall bouncing, CUDA
- **Performance**: Optimized native code generation
- **Output**: Compilation details, capabilities

## Web Interface Features

- ✅ **Code Display**: Syntax-highlighted Elixir
- ✅ **Interactive Execution**: Click-to-run
- ✅ **Console Output**: Simulated results
- ✅ **Performance Notes**: Benchmark links
- ✅ **Responsive**: Desktop/mobile

## Technical Details

- **Frontend**: HTML/CSS/JS (no build)
- **Styling**: Modern gradients, particles
- **Highlighting**: Prism.js Elixir support

## Future Enhancements

1. **Backend Integration**: Live Elixir execution
2. **Real-time Perf**: Show execution times
3. **Visualization**: Particle rendering
4. **Controls**: Adjustable params/steps

## Troubleshooting

- Check `examples/` files exist
- Open `examples.html` from root
- Browser console for JS errors

Explore examples to see fast DSL compilation and efficient runtime performance!