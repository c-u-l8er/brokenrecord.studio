defmodule BrokenRecord.Bench do
  @moduledoc """
  Comprehensive benchmarks for BrokenRecord Zero physics engine.

  Run with: mix run benchmarks/broken_record_bench.exs
  """

  # Import Benchee for benchmarking
  import Benchee

  def run_all do
    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("BrokenRecord Zero - Comprehensive Benchmarks")
    IO.puts(String.duplicate("=", 70) <> "\n")

    # Run all benchmark suites
    particle_creation_benchmarks()
    physics_calculation_benchmarks()
    collision_detection_benchmarks()
    memory_layout_benchmarks()
    scalability_benchmarks()
    conservation_benchmarks()

    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("All benchmarks complete!")
    IO.puts(String.duplicate("=", 70) <> "\n")
  end

  def particle_creation_benchmarks do
    IO.puts("Particle Creation Benchmarks")
    IO.puts("-" <> String.duplicate("-", 40))

    particles_100 = create_particles(100)
    particles_1000 = create_particles(1_000)
    particles_10000 = create_particles(10_000)

    Benchee.run(%{
      "100 particles" => fn -> create_particles(100) end,
      "1,000 particles" => fn -> create_particles(1_000) end,
      "10,000 particles" => fn -> create_particles(10_000) end
    },
    memory_time: 2,
    print: [configuration: false],
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true},
      {Benchee.Formatters.HTML, file: "benchmarks/particle_creation.html"}
    ])

    IO.puts("")
  end

  def physics_calculation_benchmarks do
    IO.puts("Physics Calculation Benchmarks")
    IO.puts("-" <> String.duplicate("-", 40))

    particles_100 = create_particles(100)
    particles_1000 = create_particles(1_000)
    particles_10000 = create_particles(10_000)

    Benchee.run(%{
      "100 particles (1 step)" => fn -> simulate_step(particles_100, 0.01) end,
      "1,000 particles (1 step)" => fn -> simulate_step(particles_1000, 0.01) end,
      "10,000 particles (1 step)" => fn -> simulate_step(particles_10000, 0.01) end
    },
    memory_time: 2,
    print: [configuration: false],
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true},
      {Benchee.Formatters.HTML, file: "benchmarks/physics_calculation.html"}
    ])

    IO.puts("")
  end

  def collision_detection_benchmarks do
    IO.puts("Collision Detection Benchmarks")
    IO.puts("-" <> String.duplicate("-", 40))

    particles_50 = create_particles_with_radius(50)
    particles_100 = create_particles_with_radius(100)
    particles_200 = create_particles_with_radius(200)

    Benchee.run(%{
      "50 particles (O(N²))" => fn -> detect_collisions(particles_50) end,
      "100 particles (O(N²))" => fn -> detect_collisions(particles_100) end,
      "200 particles (O(N²))" => fn -> detect_collisions(particles_200) end
    },
    memory_time: 2,
    print: [configuration: false],
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true},
      {Benchee.Formatters.HTML, file: "benchmarks/collision_detection.html"}
    ])

    IO.puts("")
  end

  def memory_layout_benchmarks do
    IO.puts("Memory Layout Benchmarks")
    IO.puts("-" <> String.duplicate("-", 40))

    particles = create_particles(10_000)

    Benchee.run(%{
      "AOS (Array of Structures)" => fn -> pack_aos(particles) end,
      "SOA (Structure of Arrays)" => fn -> pack_soa(particles) end
    },
    memory_time: 2,
    print: [configuration: false],
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true},
      {Benchee.Formatters.HTML, file: "benchmarks/memory_layout.html"}
    ])

    IO.puts("")
  end

  def scalability_benchmarks do
    IO.puts("Scalability Benchmarks")
    IO.puts("-" <> String.duplicate("-", 40))

    counts = [100, 500, 1_000, 5_000, 10_000]
    benchmarks = Enum.reduce(counts, %{}, fn count, acc ->
      particles = create_particles(count)
      Map.put(acc, "#{count} particles", fn -> simulate_step(particles, 0.01) end)
    end)

    Benchee.run(benchmarks,
      memory_time: 2,
      print: [configuration: false],
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true},
        {Benchee.Formatters.HTML, file: "benchmarks/scalability.html"}
      ])

    IO.puts("")
  end

  def conservation_benchmarks do
    IO.puts("Conservation Law Benchmarks")
    IO.puts("-" <> String.duplicate("-", 40))

    # Two-particle collision test
    collision_state = two_particle_collision()

    # N-body gravity test
    gravity_state = create_solar_system()

    Benchee.run(%{
      "Two-particle collision" => fn ->
        initial_energy = total_kinetic_energy(collision_state)
        result = simulate_step(collision_state.particles, 0.01)
        final_energy = total_kinetic_energy(%{particles: result})
        {initial_energy, final_energy}
      end,
      "Solar system (4 bodies)" => fn ->
        initial_energy = total_energy(gravity_state)
        result = simulate_step(gravity_state.bodies, 0.01)
        final_energy = total_energy(%{bodies: result})
        {initial_energy, final_energy}
      end
    },
    memory_time: 2,
    print: [configuration: false],
    formatters: [
      {Benchee.Formatters.Console, extended_statistics: true},
      {Benchee.Formatters.HTML, file: "benchmarks/conservation.html"}
    ])

    IO.puts("")
  end

  # Helper functions for benchmarking

  defp create_particles(n) do
    for i <- 1..n do
      %{
        position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
        velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
        mass: 1.0
      }
    end
  end

  defp create_particles_with_radius(n) do
    for i <- 1..n do
      %{
        position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
        velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
        mass: 1.0,
        radius: 0.5 + :rand.uniform() * 1.0
      }
    end
  end

  defp simulate_step(particles, dt) do
    Enum.map(particles, fn p ->
      {x, y, z} = p.position
      {vx, vy, vz} = p.velocity

      # Simple Euler integration
      new_pos = {
        x + vx * dt,
        y + vy * dt,
        z + vz * dt
      }

      # Apply gravity
      new_vel = {
        vx,
        vy,
        vz + (-9.81) * dt
      }

      %{p | position: new_pos, velocity: new_vel}
    end)
  end

  defp detect_collisions(particles) do
    Enum.reduce(particles, {[], particles}, fn p1, {collisions, rest} ->
      new_collisions = Enum.reduce(rest, [], fn p2, acc ->
        distance = particle_distance(p1, p2)
        if distance < (p1.radius || 0.5) + (p2.radius || 0.5) do
          [{p1, p2} | acc]
        else
          acc
        end
      end)
      {new_collisions ++ collisions, tl(rest)}
    end)
    |> elem(0)
  end

  defp particle_distance(p1, p2) do
    {x1, y1, z1} = p1.position
    {x2, y2, z2} = p2.position

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    :math.sqrt(dx*dx + dy*dy + dz*dz)
  end

  defp pack_aos(particles) do
    Enum.map(particles, fn p ->
      {x, y, z} = p.position
      {vx, vy, vz} = p.velocity
      [x, y, z, vx, vy, vz, p.mass]
    end)
    |> List.flatten()
  end

  defp pack_soa(particles) do
    pos_x = Enum.map(particles, fn p -> elem(p.position, 0) end)
    pos_y = Enum.map(particles, fn p -> elem(p.position, 1) end)
    pos_z = Enum.map(particles, fn p -> elem(p.position, 2) end)
    vel_x = Enum.map(particles, fn p -> elem(p.velocity, 0) end)
    vel_y = Enum.map(particles, fn p -> elem(p.velocity, 1) end)
    vel_z = Enum.map(particles, fn p -> elem(p.velocity, 2) end)
    mass = Enum.map(particles, fn p -> p.mass end)

    %{pos_x: pos_x, pos_y: pos_y, pos_z: pos_z,
      vel_x: vel_x, vel_y: vel_y, vel_z: vel_z,
      mass: mass}
  end

  defp total_kinetic_energy(state) do
    Enum.reduce(state.particles, 0.0, fn particle, acc ->
      {vx, vy, vz} = particle.velocity
      v_sq = vx*vx + vy*vy + vz*vz
      acc + 0.5 * particle.mass * v_sq
    end)
  end

  defp total_energy(state) do
    bodies = state.bodies

    # Kinetic energy
    kinetic = Enum.reduce(bodies, 0.0, fn body, acc ->
      {vx, vy, vz} = body.velocity
      v_sq = vx*vx + vy*vy + vz*vz
      acc + 0.5 * body.mass * v_sq
    end)

    # Potential energy (simplified)
    potential = Enum.reduce(bodies, 0.0, fn body1, acc1 ->
      Enum.reduce(bodies, acc1, fn body2, acc2 ->
        if body1 != body2 do
          {x1, y1, z1} = body1.position
          {x2, y2, z2} = body2.position

          dx = x1 - x2
          dy = y1 - y2
          dz = z1 - z2
          dist = :math.sqrt(dx*dx + dy*dy + dz*dz)

          if dist > 0.0 do
            acc2 - 1.0 * body1.mass * body2.mass / dist
          else
            acc2
          end
        else
          acc2
        end
      end)
    end) / 2.0

    kinetic + potential
  end

  defp two_particle_collision do
    particles = [
      %{
        position: {-5.0, 0.0, 0.0},
        velocity: {5.0, 0.0, 0.0},
        mass: 1.0,
        radius: 1.0
      },
      %{
        position: {5.0, 0.0, 0.0},
        velocity: {-5.0, 0.0, 0.0},
        mass: 1.0,
        radius: 1.0
      }
    ]
    %{particles: particles}
  end

  defp create_solar_system do
    bodies = [
      %{
        position: {0.0, 0.0, 0.0},
        velocity: {0.0, 0.0, 0.0},
        mass: 1000.0,
        radius: 5.0
      },
      %{
        position: {50.0, 0.0, 0.0},
        velocity: {0.0, 4.5, 0.0},
        mass: 1.0,
        radius: 1.0
      },
      %{
        position: {80.0, 0.0, 0.0},
        velocity: {0.0, 3.5, 0.0},
        mass: 0.5,
        radius: 0.8
      },
      %{
        position: {150.0, 0.0, 0.0},
        velocity: {0.0, 2.5, 0.0},
        mass: 10.0,
        radius: 3.0
      }
    ]
    %{bodies: bodies}
  end
end

# Main entry point
IO.puts("Starting BrokenRecord Zero benchmarks...")
BrokenRecord.Bench.run_all()
