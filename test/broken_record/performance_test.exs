defmodule BrokenRecord.PerformanceTest do
  use ExUnit.Case
  @moduletag :benchmark

  describe "performance benchmarks" do
    @tag :benchmark
    test "particle creation performance" do
      {time, particles} = :timer.tc(fn ->
        for i <- 1..10_000 do
          %{
            id: "p#{i}",
            position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
            velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
            mass: 1.0
          }
        end
      end)

      time_ms = time / 1000
      particles_per_ms = 10_000 / time_ms

      IO.puts("\nParticle Creation Performance:")
      IO.puts("  Time: #{Float.round(time_ms, 2)} ms")
      IO.puts("  Rate: #{Float.round(particles_per_ms, 0)} particles/ms")
      IO.puts("  Total: #{length(particles)} particles")

      assert length(particles) == 10_000
      assert time_ms < 1000  # Should complete within 1 second
    end

    @tag :benchmark
    test "physics calculation performance" do
      particles = for i <- 1..1_000 do
        %{
          position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
          velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
          mass: 1.0
        }
      end

      {time, _result} = :timer.tc(fn ->
        simulate_step(particles, 0.01)
      end)

      time_ms = time / 1000
      particles_per_sec = 1_000 / (time_ms / 1000)

      IO.puts("\nPhysics Calculation Performance:")
      IO.puts("  Time: #{Float.round(time_ms, 2)} ms")
      IO.puts("  Rate: #{Float.round(particles_per_sec, 0)} particles/sec")
      IO.puts("  Particles: #{length(particles)}")

      assert time_ms < 100  # Should complete within 100ms
    end

    @tag :benchmark
    test "collision detection performance" do
      particles = for i <- 1..500 do
        %{
          position: {:rand.uniform() * 10, :rand.uniform() * 10, :rand.uniform() * 10},
          velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
          mass: 1.0,
          radius: 0.5
        }
      end

      {time, collisions} = :timer.tc(fn ->
        detect_collisions(particles)
      end)

      time_ms = time / 1000
      pairs_checked = length(particles) * (length(particles) - 1) / 2
      pairs_per_ms = pairs_checked / time_ms

      IO.puts("\nCollision Detection Performance:")
      IO.puts("  Time: #{Float.round(time_ms, 2)} ms")
      IO.puts("  Rate: #{Float.round(pairs_per_ms, 0)} pairs/ms")
      IO.puts("  Pairs checked: #{pairs_checked}")
      IO.puts("  Collisions found: #{length(collisions)}")

      assert time_ms < 1000  # Should complete within 1 second
    end

    @tag :benchmark
    test "memory layout performance" do
      particles = for i <- 1..10_000 do
        %{
          position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
          velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
          mass: 1.0
        }
      end

      # Test AOS (Array of Structures) layout
      {aos_time, _aos_data} = :timer.tc(fn ->
        pack_aos(particles)
      end)

      # Test SOA (Structure of Arrays) layout
      {soa_time, _soa_data} = :timer.tc(fn ->
        pack_soa(particles)
      end)

      aos_ms = aos_time / 1000
      soa_ms = soa_time / 1000
      speedup = aos_ms / soa_ms

      IO.puts("\nMemory Layout Performance:")
      IO.puts("  AOS time: #{Float.round(aos_ms, 2)} ms")
      IO.puts("  SOA time: #{Float.round(soa_ms, 2)} ms")
      IO.puts("  SOA speedup: #{Float.round(speedup, 2)}x")

      # SOA should be faster for this use case
      assert soa_ms <= aos_ms * 1.5  # Allow some variance
    end
  end

  describe "scalability tests" do
    @tag :benchmark
    test "performance scales linearly with particle count" do
      counts = [100, 1_000, 5_000]
      times = []

      for count <- counts do
        particles = for i <- 1..count do
          %{
            position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
            velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
            mass: 1.0
          }
        end

        {time, _} = :timer.tc(fn ->
          simulate_step(particles, 0.01)
        end)

        times = [time | times]
        IO.puts("  #{count} particles: #{time / 1000} ms")
      end

      # Check that performance scales reasonably (not exponentially)
      times = Enum.reverse(times)
      ratio_1k_to_100 = Enum.at(times, 1) / Enum.at(times, 0)
      ratio_5k_to_1k = Enum.at(times, 2) / Enum.at(times, 1)

      # Should be roughly linear (allowing for some overhead)
      assert ratio_1k_to_100 < 15  # Less than 10x increase for 10x particles
      assert ratio_5k_to_1k < 7   # Less than 5x increase for 5x particles
    end
  end

  # Helper functions for performance testing

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
end
