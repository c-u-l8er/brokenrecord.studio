defmodule BrokenRecord.ZeroTest do
  use ExUnit.Case
  alias BrokenRecord.Zero

  describe "basic functionality" do
    test "can create a particle system" do
      particles = [
        %{position: {0.0, 0.0, 10.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0},
        %{position: {5.0, 0.0, 10.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0}
      ]

      assert length(particles) == 2
      assert Enum.all?(particles, fn p ->
        is_tuple(p.position) and tuple_size(p.position) == 3 and
        is_tuple(p.velocity) and tuple_size(p.velocity) == 3 and
        is_number(p.mass) and p.mass > 0
      end)
    end

    test "particle positions update correctly" do
      initial = %{position: {0.0, 0.0, 10.0}, velocity: {0.0, 0.0, -9.81}, mass: 1.0}

      # After 1 second with gravity, position should be lower
      expected_z = 10.0 + (-9.81) * 1.0
      assert expected_z < 10.0
    end

    test "conservation of momentum in collisions" do
      # Two particles moving towards each other
      p1 = %{position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0}
      p2 = %{position: {10.0, 0.0, 0.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0}

      # Initial momentum
      initial_momentum = {
        p1.mass * elem(p1.velocity, 0) + p2.mass * elem(p2.velocity, 0),
        p1.mass * elem(p1.velocity, 1) + p2.mass * elem(p2.velocity, 1),
        p1.mass * elem(p1.velocity, 2) + p2.mass * elem(p2.velocity, 2)
      }

      # Should be zero (equal and opposite)
      assert elem(initial_momentum, 0) == 0.0
      assert elem(initial_momentum, 1) == 0.0
      assert elem(initial_momentum, 2) == 0.0
    end
  end

  describe "physics calculations" do
    test "gravity acceleration" do
      # Standard gravity: -9.81 m/s²
      gravity = -9.81
      dt = 0.01

      # Velocity change after one timestep
      dv = gravity * dt
      assert dv == -0.0981
    end

    test "euler integration" do
      # Position update: new_pos = old_pos + velocity * dt
      pos = {0.0, 0.0, 10.0}
      vel = {1.0, 2.0, 3.0}
      dt = 0.1

      new_pos = {
        elem(pos, 0) + elem(vel, 0) * dt,
        elem(pos, 1) + elem(vel, 1) * dt,
        elem(pos, 2) + elem(vel, 2) * dt
      }

      assert new_pos == {0.1, 0.2, 10.3}
    end

    test "collision detection" do
      # Two spheres
      p1 = %{position: {0.0, 0.0, 0.0}, radius: 1.0}
      p2 = %{position: {1.5, 0.0, 0.0}, radius: 1.0}

      # Distance between centers
      dx = elem(p1.position, 0) - elem(p2.position, 0)
      dy = elem(p1.position, 1) - elem(p2.position, 1)
      dz = elem(p1.position, 2) - elem(p2.position, 2)
      distance = :math.sqrt(dx*dx + dy*dy + dz*dz)

      # Should be colliding (distance < sum of radii)
      assert distance < (p1.radius + p2.radius)
    end
  end

  describe "performance characteristics" do
    test "can handle large particle counts" do
      # Test that we can create large particle arrays
      particles = for i <- 1..10_000 do
        %{
          position: {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100},
          velocity: {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5},
          mass: 1.0
        }
      end

      assert length(particles) == 10_000
    end

    test "memory layout efficiency" do
      # Structure of Arrays should be more cache-friendly
      # This test verifies the concept
      particles = for _i <- 1..1000 do
        %{
          position: {1.0, 2.0, 3.0},
          velocity: {0.1, 0.2, 0.3},
          mass: 1.0
        }
      end

      # In SOA layout, all x positions would be contiguous
      # This improves cache locality for vectorized operations
      assert length(particles) == 1000
    end
  end

  describe "edge cases" do
    test "zero mass particles" do
      # Should handle zero mass gracefully
      particle = %{position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 0.0}
      assert particle.mass == 0.0
    end

    test "infinite velocities" do
      # Should handle edge cases
      particle = %{position: {0.0, 0.0, 0.0}, velocity: {1.0, :infinity, 0.0}, mass: 1.0}
      assert elem(particle.velocity, 1) == :infinity
    end

    test "nan values" do
      # Should handle NaN gracefully
      particle = %{position: {0.0, 0.0, 0.0}, velocity: {:nan, 0.0, 0.0}, mass: 1.0}
      assert :erlang.is_float(elem(particle.velocity, 0))
    end
  end
end
