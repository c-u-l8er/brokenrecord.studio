defmodule BenchmarkAII do
  @moduledoc """
  Comprehensive benchmark suite for AII core systems using Benchee.

  Tests the performance of different AII DSL implementations across various
  physics domains and system scales. Measures execution time, memory usage,
  and provides detailed statistical analysis.
  """

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

    # Define test systems inline
    defmodule BenchParticleSystem do
      use AII.DSL

      defagent Particle do
        state :position, AII.Types.Vec3
        state :velocity, AII.Types.Vec3
        property :mass, Float, invariant: true
      end

      definteraction :gravity, accelerator: :cpu do
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

      definteraction :integrate, accelerator: :cpu do
        let p do
          p.position = p.position + p.velocity * 0.01
        end
      end
    end

    # Create particle sets
    particle_sets = %{
      "10 particles" => create_particles(10),
      "50 particles" => create_particles(50),
      "100 particles" => create_particles(100)
    }

    Benchee.run(%{
      "Particle Physics - 10 particles" => fn ->
        run_simulation(BenchParticleSystem, particle_sets["10 particles"], 5)
      end,
      "Particle Physics - 50 particles" => fn ->
        run_simulation(BenchParticleSystem, particle_sets["50 particles"], 5)
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

    defmodule BenchGravitySystem do
      use AII.DSL

      defagent Body do
        state :position, AII.Types.Vec3
        state :velocity, AII.Types.Vec3
        property :mass, Float, invariant: true
      end

      definteraction :gravitational_force, accelerator: :cpu do
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

      definteraction :integrate_motion, accelerator: :cpu do
        let b do
          b.position = b.position + b.velocity * 0.01
        end
      end
    end

    solar_system = [
      %{position: {0.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.989e30, energy: 0.0, particle_id: 1},  # Sun
      %{position: {1.496e11, 0.0, 0.0}, velocity: {0.0, 2.978e4, 0.0}, mass: 5.972e24, energy: 0.0, particle_id: 2},  # Earth
      %{position: {2.279e11, 0.0, 0.0}, velocity: {0.0, 2.413e4, 0.0}, mass: 6.39e23, energy: 0.0, particle_id: 3},   # Mars
      %{position: {5.791e11, 0.0, 0.0}, velocity: {0.0, 1.307e4, 0.0}, mass: 1.898e27, energy: 0.0, particle_id: 4},  # Jupiter
    ]

    Benchee.run(%{
      "Solar System (4 bodies)" => fn ->
        run_simulation(BenchGravitySystem, solar_system, 3)
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

      definteraction :diffusion, accelerator: :cpu do
        let m do
          random_force = {(:rand.uniform() - 0.5) * 0.1, (:rand.uniform() - 0.5) * 0.1, (:rand.uniform() - 0.5) * 0.1}
          m.velocity = m.velocity + random_force
          m.position = m.position + m.velocity * 0.01
        end
      end

      definteraction :reaction, accelerator: :cpu do
        let {a, b} when a.type == :A and b.type == :B do
          if distance(a.position, b.position) < 5.0 do
            new_mass = a.mass + b.mass
            new_energy = a.energy + b.energy + 10.0
            midpoint = (a.position + b.position) * 0.5
            a.position = midpoint
            a.mass = new_mass
            a.energy = new_energy
            a.type = :AB
            b.position = {0.0, 0.0, 0.0}
            b.mass = 0.0
            b.energy = 0.0
          end
        end
      end

      definteraction :conserve_quantities, accelerator: :cpu do
        let m do
          conserved(:energy, m.energy)
          conserved(:mass, m.mass)
        end
      end
    end

    chemical_particles = [
      %{position: {0.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.0, energy: 5.0, type: :A, particle_id: 1},
      %{position: {3.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.0, energy: 5.0, type: :B, particle_id: 2},
      %{position: {10.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.0, energy: 5.0, type: :A, particle_id: 3},
      %{position: {13.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 1.0, energy: 5.0, type: :B, particle_id: 4},
    ]

    Benchee.run(%{
      "Chemical Reactions (4 molecules)" => fn ->
        run_simulation(BenchChemicalSystem, chemical_particles, 5)
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

    defmodule BenchScalabilitySystem do
      use AII.DSL

      defagent Particle do
        state :position, AII.Types.Vec3
        state :velocity, AII.Types.Vec3
        property :mass, Float, invariant: true
      end

      definteraction :simple_update, accelerator: :cpu do
        let p do
          p.velocity = p.velocity * 0.99  # Damping
          p.position = p.position + p.velocity * 0.01
        end
      end
    end

    scales = [10, 50, 100, 200]

    scales = [10, 50]  # Reduced scales

    benchmarks = Enum.reduce(scales, %{}, fn scale, acc ->
      particles = create_particles(scale)
      Map.put(acc, "#{scale} particles", fn ->
        run_simulation(BenchScalabilitySystem, particles, 3)
      end)
    end)

    Benchee.run(benchmarks,
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

    defmodule BenchConservationSystem do
      use AII.DSL

      conserved_quantity :energy, type: :scalar, law: :sum
      conserved_quantity :momentum, type: :vector3, law: :sum

      defagent ConservedParticle do
        state :position, AII.Types.Vec3
        state :velocity, AII.Types.Vec3
        property :mass, Float, invariant: true
        state :energy, AII.Types.Conserved
        state :momentum, AII.Types.Conserved
      end

      definteraction :elastic_collision, accelerator: :cpu do
        let {p1, p2} do
          if distance(p1.position, p2.position) < 2.0 do
            temp_vel = p1.velocity
            p1.velocity = p2.velocity
            p2.velocity = temp_vel
          end
        end
      end

      definteraction :integrate_with_conservation, accelerator: :cpu do
        let p do
          p.position = p.position + p.velocity * 0.01
          kinetic_energy = 0.5 * p.mass * magnitude_squared(p.velocity)
          p.energy = conserved(p.energy, kinetic_energy)
          p.momentum = conserved(p.momentum, p.velocity * p.mass)
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

      definteraction :elastic_collision, accelerator: :cpu do
        let {p1, p2} do
          if distance(p1.position, p2.position) < 2.0 do
            temp_vel = p1.velocity
            p1.velocity = p2.velocity
            p2.velocity = temp_vel
          end
        end
      end

      definteraction :integrate_simple, accelerator: :cpu do
        let p do
          p.position = p.position + p.velocity * 0.01
        end
      end
    end

    particles = create_particles(50)

    Benchee.run(%{
      "With Conservation Laws" => fn ->
        run_simulation(BenchConservationSystem, particles, 5)
      end,
      "Without Conservation Laws" => fn ->
        run_simulation(BenchNoConservationSystem, particles, 5)
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
