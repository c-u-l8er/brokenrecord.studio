defmodule ParticlePhysicsBenchmark do
  @moduledoc """
  Dedicated benchmark for particle physics simulations using N-body gravity.

  This benchmark tests the performance of gravitational N-body simulations
  with varying numbers of particles, measuring execution time and memory usage.
  """

  def run do
    IO.puts("🚀 Particle Physics N-Body Gravity Benchmark")
    IO.puts("==============================================")

    # Define particle counts to test
    particle_counts = [10, 50, 100, 500]

    # Run benchmarks for each count
    Enum.each(particle_counts, fn count ->
      IO.puts("\n🔬 Benchmarking #{count} particles...")

      Benchee.run(%{
        "#{count} particles - 100 steps" => fn ->
          run_nbody_simulation(count, 100)
        end,
        "#{count} particles - 500 steps" => fn ->
          run_nbody_simulation(count, 500)
        end
      },
      time: 2,
      memory_time: 1,
      parallel: 1,
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true},
        {Benchee.Formatters.HTML, file: "benchmarks/particle_physics_benchmark_#{count}_particles.html"}
      ])
    end)

    IO.puts("\n✅ Particle physics benchmarks complete!")
    IO.puts("📊 Check benchmarks/particle_physics_benchmark_*.html for detailed results")
  end

  # ============================================================================
  # N-Body Gravity Simulation System
  # ============================================================================

  defmodule NBodySystem do
    use AII.DSL

    defagent Particle do
      state :position, AII.Types.Vec3
      state :velocity, AII.Types.Vec3
      property :mass, Float, invariant: true
    end

    definteraction :gravitational_force, accelerator: :auto do
      let {p1, p2} do
        r_vec = p2.position - p1.position
        r_sq = dot(r_vec, r_vec)
        r = sqrt(r_sq)

        if r > 1.0e-6 do  # Avoid division by zero
          force_magnitude = 6.67430e-11 * p1.mass * p2.mass / r_sq
          force_direction = normalize(r_vec)

          # Update velocities (symplectic Euler)
          dt = 0.01
          p1.velocity = p1.velocity + force_direction * (force_magnitude * dt / p1.mass)
          p2.velocity = p2.velocity - force_direction * (force_magnitude * dt / p2.mass)
        end
      end
    end

    definteraction :integrate_position, accelerator: :auto do
      let p do
        dt = 0.01
        p.position = p.position + p.velocity * dt
      end
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp run_nbody_simulation(num_particles, steps) do
    particles = create_random_particles(num_particles)
    AII.run_simulation(NBodySystem, steps: steps, dt: 0.01, particles: particles)
  end

  defp create_random_particles(count) do
    Enum.map(1..count, fn i ->
      # Random position in a cube of side 100 units
      pos_x = :rand.uniform() * 100.0 - 50.0
      pos_y = :rand.uniform() * 100.0 - 50.0
      pos_z = :rand.uniform() * 100.0 - 50.0

      # Random velocity, small to avoid immediate escape
      vel_x = (:rand.uniform() - 0.5) * 2.0
      vel_y = (:rand.uniform() - 0.5) * 2.0
      vel_z = (:rand.uniform() - 0.5) * 2.0

      # Mass between 0.1 and 10 solar masses
      mass = :rand.uniform() * 9.9 + 0.1

      %{
        position: {pos_x, pos_y, pos_z},
        velocity: {vel_x, vel_y, vel_z},
        mass: mass,
        energy: 0.0,
        particle_id: i
      }
    end)
  end

  # Vector math helpers (same as in benchmark_aii.exs)
  defp magnitude({x, y, z}), do: :math.sqrt(x*x + y*y + z*z)
  defp dot({x1, y1, z1}, {x2, y2, z2}), do: x1*x2 + y1*y2 + z1*z2
  defp normalize(vec), do: div_vector(vec, magnitude(vec))
  defp div_vector({x, y, z}, s), do: {x/s, y/s, z/s}
  defp mul_vector({x, y, z}, s), do: {x*s, y*s, z*s}
  defp add_vector({x1, y1, z1}, {x2, y2, z2}), do: {x1+x2, y1+y2, z1+z2}
  defp sub_vector({x1, y1, z1}, {x2, y2, z2}), do: {x1-x2, y1-y2, z1-z2}
  defp sqrt(x), do: :math.sqrt(x)
end

# Run the benchmark
ParticlePhysicsBenchmark.run()
