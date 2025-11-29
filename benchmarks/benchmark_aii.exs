defmodule BenchmarkAII do
  @moduledoc """
  Comprehensive benchmark suite for AII core systems using Benchee.

  Tests the performance of different AII DSL implementations across various
  physics domains and system scales. Measures execution time, memory usage,
  and provides detailed statistical analysis.
  """

  # ============================================================================
  # Benchmark System Modules (Pre-compiled)
  # ============================================================================

  defmodule BenchParticleSystem do
    use AII.DSL

    defagent Particle do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      property :mass, Float, invariant: true
    end

    definteraction :gravity, accelerator: :auto do
      let {p1, p2} do
        r_vec = p2.position - p1.position
        r = magnitude(r_vec)
        if r > 0.0 do
          force = 6.67e-11 * p1.mass * p2.mass / (r * r)
          dir = normalize(r_vec)
          p1.velocity = p1.velocity + dir * force * 0.01 / p1.mass
          p2.velocity = p2.velocity - dir * force * 0.01 / p2.mass
        end
      end
    end

    definteraction :integrate, accelerator: :auto do
      let p do
        p.position = p.position + p.velocity * 0.01
      end
    end

    definteraction :detect_collisions, accelerator: :auto do
      let {p1, p2} do
        # Simple collision detection using RT cores
        distance = magnitude(p2.position - p1.position)
        if distance < 2.0 do
          # Collision detected - could apply collision response
          # For now, just mark that collision occurred
          conserved(:collisions, 1)
        end
      end
    end
  end

  defmodule BenchGravitySystem do
    use AII.DSL

    defagent Body do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      property :mass, Float, invariant: true
    end

    definteraction :gravitational_force, accelerator: :auto do
      let {b1, b2} do
        r_vec = b2.position - b1.position
        r_sq = dot(r_vec, r_vec)
        r = sqrt(r_sq)

        if r > 1.0 do
          force_magnitude = 6.67e-11 * b1.mass * b2.mass / r_sq
          force_direction = normalize(r_vec)

          b1.velocity = b1.velocity + force_direction * force_magnitude * 0.001 / b1.mass
          b2.velocity = b2.velocity - force_direction * force_magnitude * 0.001 / b2.mass
        end
      end
    end

    definteraction :integrate_motion, accelerator: :auto do
      let b do
        b.position = b.position + b.velocity * 0.01
      end
    end

    definteraction :gravitational_collisions, accelerator: :auto do
      let {b1, b2} do
        distance = magnitude(b2.position - b1.position)
        if distance < 5.0 do  # Larger collision radius for celestial bodies
          # Elastic collision response (simplified)
          relative_velocity = b2.velocity - b1.velocity
          # Apply collision impulse
          impulse = relative_velocity * 0.5  # Simplified coefficient
          b1.velocity = b1.velocity + impulse / b1.mass
          b2.velocity = b2.velocity - impulse / b2.mass
        end
      end
    end
  end

  defmodule BenchChemicalSystem do
    use AII.DSL

    conserved_quantity :energy, type: :scalar, law: :sum
    conserved_quantity :mass, type: :scalar, law: :sum

    defagent Molecule do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      property :mass, Float, invariant: true
      property :energy, Float
      property :type, Atom, invariant: true
    end

    definteraction :diffusion, accelerator: :auto do
      let m do
        random_force = {(:rand.uniform() - 0.5) * 0.1, (:rand.uniform() - 0.5) * 0.1, (:rand.uniform() - 0.5) * 0.1}
        m.velocity = m.velocity + random_force
        m.position = m.position + m.velocity * 0.01
      end
    end

    definteraction :reaction, accelerator: :auto do
      let {a, b} when a.type == :A and b.type == :B do
        if distance(a.position, b.position) < 5.0 do
          new_mass = a.mass + b.mass
          new_energy = a.energy + b.energy + 50.0  # Exothermic reaction
          new_position = midpoint(a.position, b.position)

          # Create new molecule, remove reactants
          # (Simplified - would need proper reaction mechanics)
        end
      end
    end

    definteraction :conserve_quantities, accelerator: :auto do
      let m do
        conserved(:energy, m.energy)
        conserved(:mass, m.mass)
      end
    end
  end

  defmodule BenchScalabilitySystem do
    use AII.DSL

    defagent Particle do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      property :mass, Float, invariant: true
    end

    definteraction :simple_update, accelerator: :auto do
      let p do
        p.velocity = p.velocity * 0.99  # Damping
        p.position = p.position + p.velocity * 0.01
      end
    end
  end

  defmodule BenchConservationSystem do
    use AII.DSL

    conserved_quantity :energy, type: :scalar, law: :sum
    conserved_quantity :momentum, type: :vector3, law: :sum

    defagent Particle do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      state :momentum, AII.Types.Conserved
      property :mass, Float, invariant: true
    end

    definteraction :elastic_collision, accelerator: :auto do
      let {p1, p2} do
        if distance(p1.position, p2.position) < 2.0 do
          # Simple elastic collision
          p1.velocity = p2.velocity
          p2.velocity = p1.velocity
        end
      end
    end

    definteraction :integrate_with_conservation, accelerator: :auto do
      let p do
        p.position = p.position + p.velocity * 0.01
        conserved(:energy, 0.5 * p.mass * magnitude(p.velocity) ** 2)
        conserved(:momentum, p.velocity * p.mass)
      end
    end

    definteraction :conserved_collisions, accelerator: :auto do
      let {p1, p2} do
        distance = magnitude(p2.position - p1.position)
        if distance < 2.0 do
          # Elastic collision with conservation
          total_energy_before = 0.5 * p1.mass * magnitude(p1.velocity)**2 +
                               0.5 * p2.mass * magnitude(p2.velocity)**2
          total_momentum_before = p1.velocity * p1.mass + p2.velocity * p2.mass

          # Simple elastic collision (swap velocities for equal mass)
          temp_velocity = p1.velocity
          p1.velocity = p2.velocity
          p2.velocity = temp_velocity

          # Verify conservation
          conserved(:energy, 0.5 * p1.mass * magnitude(p1.velocity)**2 +
                           0.5 * p2.mass * magnitude(p2.velocity)**2)
          conserved(:momentum, p1.velocity * p1.mass + p2.velocity * p2.mass)
        end
      end
    end
  end

  defmodule BenchNoConservationSystem do
    use AII.DSL

    defagent SimpleParticle do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      property :mass, Float, invariant: true
    end

    definteraction :elastic_collision, accelerator: :auto do
      let {p1, p2} do
        if distance(p1.position, p2.position) < 2.0 do
          # Simple elastic collision
          p1.velocity = p2.velocity
          p2.velocity = p1.velocity
        end
      end
    end

    definteraction :integrate_simple, accelerator: :auto do
      let p do
        p.position = p.position + p.velocity * 0.01
      end
    end
  end

  def run do
    IO.puts("🚀 AII Core Benchmark Suite (Benchee)")
    IO.puts("=====================================")

    # Run benchmark categories
    benchmark_particle_systems()
    benchmark_gravity_systems()
    benchmark_chemical_systems()
    benchmark_scalability()
    benchmark_conservation_overhead()

    IO.puts("\n✅ AII benchmarks complete!")
    IO.puts("📊 Check benchmarks/benchmark_aii.html for detailed results")
  end

  # ============================================================================
  # Particle Physics Benchmarks
  # ============================================================================

  defp benchmark_particle_systems do
    IO.puts("\n🔬 Benchmarking Particle Physics Systems...")

    # Create particle sets
    particle_sets = %{
      "10 particles" => create_particles(10),
      "50 particles" => create_particles(50),
      "100 particles" => create_particles(100)
    }

    # Test caching by running multiple simulations in sequence
    Benchee.run(%{
      "Particle Physics - 10 particles (cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchParticleSystem, particle_sets["10 particles"], 1)
        run_simulation(BenchParticleSystem, particle_sets["10 particles"], 1)
        run_simulation(BenchParticleSystem, particle_sets["10 particles"], 1)
      end,
      "Particle Physics - 50 particles (cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchParticleSystem, particle_sets["50 particles"], 1)
        run_simulation(BenchParticleSystem, particle_sets["50 particles"], 1)
        run_simulation(BenchParticleSystem, particle_sets["50 particles"], 1)
      end
    },
    time: 2,
    memory_time: 1,
    parallel: 1,
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true}
    ])
  end

  # ============================================================================
  # Gravity System Benchmarks
  # ============================================================================

  defp benchmark_gravity_systems do
    IO.puts("\n🌍 Benchmarking Gravity Systems...")

    solar_system = [
      %{position: {0.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.989e30, energy: 0.0, particle_id: 1},  # Sun
      %{position: {1.496e11, 0.0, 0.0}, velocity: {0.0, 2.978e4, 0.0}, mass: 5.972e24, energy: 0.0, particle_id: 2},  # Earth
      %{position: {2.279e11, 0.0, 0.0}, velocity: {0.0, 2.413e4, 0.0}, mass: 6.39e23, energy: 0.0, particle_id: 3},   # Mars
      %{position: {5.791e11, 0.0, 0.0}, velocity: {0.0, 1.307e4, 0.0}, mass: 1.898e27, energy: 0.0, particle_id: 4},  # Jupiter
    ]

    Benchee.run(%{
      "Solar System (4 bodies, cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchGravitySystem, solar_system, 1)
        run_simulation(BenchGravitySystem, solar_system, 1)
        run_simulation(BenchGravitySystem, solar_system, 1)
      end
    },
    time: 2,
    memory_time: 1,
    parallel: 1,
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true}
    ])
  end

  # ============================================================================
  # Chemical Reaction Benchmarks
  # ============================================================================

  defp benchmark_chemical_systems do
    IO.puts("\n🧪 Benchmarking Chemical Reaction Systems...")

    # Create chemical system
    molecules = [
      %{position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 10.0, energy: 100.0, type: :A, particle_id: 1},
      %{position: {10.0, 0.0, 0.0}, velocity: {-1.0, 0.0, 0.0}, mass: 10.0, energy: 100.0, type: :B, particle_id: 2},
      %{position: {0.0, 10.0, 0.0}, velocity: {0.0, -1.0, 0.0}, mass: 10.0, energy: 100.0, type: :A, particle_id: 3},
      %{position: {10.0, 10.0, 0.0}, velocity: {0.0, 1.0, 0.0}, mass: 10.0, energy: 100.0, type: :B, particle_id: 4}
    ]

    Benchee.run(%{
      "Chemical Reactions (4 molecules, cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchChemicalSystem, molecules, 1)
        run_simulation(BenchChemicalSystem, molecules, 1)
        run_simulation(BenchChemicalSystem, molecules, 1)
      end
    },
    time: 2,
    memory_time: 1,
    parallel: 1,
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true}
    ])
  end

  # ============================================================================
  # Scalability Benchmarks
  # ============================================================================

  defp benchmark_scalability do
    IO.puts("\n📈 Benchmarking Scalability...")

    Benchee.run(%{
      "Scalability - 10 particles (cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchScalabilitySystem, create_particles(10), 1)
        run_simulation(BenchScalabilitySystem, create_particles(10), 1)
        run_simulation(BenchScalabilitySystem, create_particles(10), 1)
      end,
      "Scalability - 50 particles (cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchScalabilitySystem, create_particles(50), 1)
        run_simulation(BenchScalabilitySystem, create_particles(50), 1)
        run_simulation(BenchScalabilitySystem, create_particles(50), 1)
      end
    },
    time: 2,
    memory_time: 1,
    parallel: 1,
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true}
    ])
  end

  # ============================================================================
  # Conservation Overhead Benchmarks
  # ============================================================================

  defp benchmark_conservation_overhead do
    IO.puts("\n⚖️  Benchmarking Conservation Overhead...")

    # Create test particles
    particles = create_particles(10)

    Benchee.run(%{
      "With Conservation Laws (cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchConservationSystem, particles, 1)
        run_simulation(BenchConservationSystem, particles, 1)
        run_simulation(BenchConservationSystem, particles, 1)
      end,
      "Without Conservation Laws (cached)" => fn ->
        # Run multiple times to test caching
        run_simulation(BenchNoConservationSystem, particles, 1)
        run_simulation(BenchNoConservationSystem, particles, 1)
        run_simulation(BenchNoConservationSystem, particles, 1)
      end
    },
    time: 2,
    memory_time: 1,
    parallel: 1,
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true}
    ])
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp run_simulation(system_module, particles, steps) do
    AII.run_simulation(system_module, steps: steps, dt: 0.01, particles: particles)
  end

  defp create_particles(count) do
    Enum.map(1..count, fn i ->
      %{
        position: {(i * 2.0), 0.0, 0.0},
        velocity: {1.0, 0.0, 0.0},
        mass: 1.0,
        energy: 0.0,
        particle_id: i
      }
    end)
  end

  # Vector math helpers
  defp magnitude({x, y, z}), do: :math.sqrt(x*x + y*y + z*z)
  defp magnitude_squared({x, y, z}), do: x*x + y*y + z*z
  defp dot({x1, y1, z1}, {x2, y2, z2}), do: x1*x2 + y1*y2 + z1*z2
  defp normalize(vec), do: div_vector(vec, magnitude(vec))
  defp div_vector({x, y, z}, s), do: {x/s, y/s, z/s}
  defp mul_vector({x, y, z}, s), do: {x*s, y*s, z*s}
  defp add_vector({x1, y1, z1}, {x2, y2, z2}), do: {x1+x2, y1+y2, z1+z2}
  defp sub_vector({x1, y1, z1}, {x2, y2, z2}), do: {x1-x2, y1-y2, z1-z2}
  defp sqrt(x), do: :math.sqrt(x)
  defp distance(a, b), do: magnitude(sub_vector(a, b))
end

# Run the benchmark suite
BenchmarkAII.run()
