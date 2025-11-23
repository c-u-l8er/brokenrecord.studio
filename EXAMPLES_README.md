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
# Note: This example defines the system but doesn't run it by default
# See the Demo module in my_physics.ex for usage
```

## Available Examples

### 1. Actor Model (`actor_model.ex`)
- **Concept**: Actor-based concurrency using lafont interaction nets
- **Features**: Message passing, supervision trees, fault tolerance, load balancing
- **Output**: Shows actor states, message counts, and system statistics

### 2. Chemical Reaction Network (`chemical_reaction_net.ex`)
- **Concept**: Chemical reaction simulation with conservation laws
- **Features**: Synthesis reactions, decomposition, catalysis, thermal dynamics
- **Output**: Molecule counts, mass conservation verification

### 3. Gravity Simulation (`gravity_simulation.ex`)
- **Concept**: N-body gravity simulation with conservation guarantees
- **Features**: Gravitational forces, energy/momentum conservation
- **Output**: Energy levels, conservation verification, system statistics

### 4. Custom Physics System (`my_physics.ex`)
- **Concept**: Complete physics system with GPU acceleration
- **Features**: Particle collisions, wall bouncing, CUDA compilation
- **Output**: System compilation details and performance capabilities

## Web Interface Features

The `examples.html` file provides:

- ✅ **Code Display**: Syntax-highlighted Elixir code for each example
- ✅ **Interactive Execution**: Click-to-run interface with loading states
- ✅ **Console Output**: Simulated results showing what each example produces
- ✅ **Error Handling**: Graceful error display and recovery
- ✅ **Responsive Design**: Works on desktop and mobile devices
- ✅ **Search**: Find examples by name or content

## Technical Details

- **Frontend**: Pure HTML/CSS/JavaScript (no build step required)
- **Styling**: Modern gradient design with animated particles
- **Syntax Highlighting**: Prism.js for Elixir code
- **Console Simulation**: Mock outputs demonstrating real example behavior
- **Browser Support**: All modern browsers (Chrome, Firefox, Safari, Edge)

## Future Enhancements

To make this a fully functional system, you could:

1. **Backend Integration**: Connect the web interface to actual Elixir execution
2. **Real-time Output**: Stream live console output during execution
3. **Interactive Controls**: Add parameters to control simulation steps, dt, etc.
4. **Visualization**: Add graphical output for particle systems
5. **Performance Metrics**: Show execution time and memory usage

## Troubleshooting

If examples don't load:
1. Check that `examples/` directory contains the .ex files
2. Ensure you're opening `examples.html` from the project root
3. Check browser console for JavaScript errors
4. Verify file permissions if running from local filesystem

For the complete BrokenRecord.Zero experience, the web interface provides an easy way to explore and understand the physics programming language capabilities!