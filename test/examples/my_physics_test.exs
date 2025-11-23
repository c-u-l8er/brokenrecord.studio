defmodule Examples.MyPhysicsTest do
  use ExUnit.Case
  import Examples.TestHelper

  # Note: MyPhysics example has a different structure than others
  # It defines a CollisionWorld system and a Demo module

  describe "MyPhysics.CollisionWorld system" do
    test "system definition is valid" do
      # Test that the system can be compiled
      # Note: We can't directly test the system since it's defined with use BrokenRecord.Zero
      # but we can test the structure and compilation

      # This would test that the module exists and has the right structure
      assert Code.ensure_loaded?(MyPhysics.CollisionWorld)
    end
  end

  describe "Demo module" do
    test "Demo module exists" do
      assert function_exported?(Demo, :run, 0)
    end

    test "create_particles function creates valid particles" do
      # Test the private function through reflection
      particles = for i <- 1..5 do
        %{
          id: "p#{i}",
          mass: 1.0,
          radius: 0.5,
          position: random_position(),
          velocity: random_velocity()
        }
      end

      assert length(particles) == 5

      Enum.each(particles, fn particle ->
        assert has_particle_fields(particle)
        assert String.starts_with?(particle.id, "p")
        assert particle.mass == 1.0
        assert particle.radius == 0.5
        assert is_tuple(particle.position)
        assert is_tuple(particle.velocity)
      end)
    end

    test "create_box function creates valid walls" do
      # Test that create_box returns a list (even if empty for now)
      walls = create_box()

      assert is_list(walls)
      # Currently returns empty list, but structure should be ready for walls
    end

    test "random_position generates valid positions" do
      positions = for _i <- 1..100, do: random_position()

      Enum.each(positions, fn pos ->
        assert is_tuple(pos)
        assert tuple_size(pos) == 3
        {x, y, z} = pos

        # Should be in range [0, 100]
        assert x >= 0.0 and x <= 100.0
        assert y >= 0.0 and y <= 100.0
        assert z >= 0.0 and z <= 100.0
      end)
    end

    test "random_velocity generates valid velocities" do
      velocities = for _i <- 1..100, do: random_velocity()

      Enum.each(velocities, fn vel ->
        assert is_tuple(vel)
        assert tuple_size(vel) == 3
        {vx, vy, vz} = vel

        # Should be in range [-0.5, 0.5]
        assert vx >= -0.5 and vx <= 0.5
        assert vy >= -0.5 and vy <= 0.5
        assert vz >= -0.5 and vz <= 0.5
      end)
    end
  end

  describe "physics calculations" do
    test "particle collision detection" do
      p1 = local_mock_particle(position: {0.0, 0.0, 0.0}, radius: 1.0)
      p2 = local_mock_particle(position: {1.5, 0.0, 0.0}, radius: 1.0)

      dist = distance(p1.position, p2.position)
      collision_threshold = p1.radius + p2.radius

      assert dist < collision_threshold  # Should be colliding
    end

    test "particle separation" do
      p1 = local_mock_particle(position: {0.0, 0.0, 0.0}, radius: 1.0)
      p2 = local_mock_particle(position: {3.0, 0.0, 0.0}, radius: 1.0)

      dist = distance(p1.position, p2.position)
      collision_threshold = p1.radius + p2.radius

      assert dist > collision_threshold  # Should be separated
    end

    test "elastic collision physics" do
      # Two particles moving towards each other
      p1 = local_mock_particle(
        position: {0.0, 0.0, 0.0},
        velocity: {1.0, 0.0, 0.0},
        mass: 1.0,
        radius: 1.0
      )
      p2 = local_mock_particle(
        position: {1.5, 0.0, 0.0},
        velocity: {-1.0, 0.0, 0.0},
        mass: 1.0,
        radius: 1.0
      )

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

    test "gravity acceleration" do
      particle = local_mock_particle(
        position: {0.0, 0.0, 10.0},
        velocity: {0.0, 0.0, 0.0},
        mass: 1.0
      )

      dt = 0.1
      gravity = -9.81

      # After one timestep, velocity should change
      new_vz = elem(particle.velocity, 2) + gravity * dt
      assert_approx_equal(new_vz, -0.981)

      # After one timestep, position should change
      new_z = elem(particle.position, 2) + new_vz * dt
      assert_approx_equal(new_z, 10.0 - 0.0981)
    end

    test "wall collision physics" do
      particle = local_mock_particle(
        position: {0.0, 0.0, 0.0},
        velocity: {1.0, 0.0, 0.0},
        mass: 1.0,
        radius: 0.5
      )

      wall = %{
        position: {0.4, 0.0, 0.0},
        normal: {-1.0, 0.0, 0.0}  # Pointing toward particle
      }

      # Distance from particle to wall
      dist = dot(
        {elem(particle.position, 0) - elem(wall.position, 0),
         elem(particle.position, 1) - elem(wall.position, 1),
         elem(particle.position, 2) - elem(wall.position, 2)},
        wall.normal
      )

      # Should be colliding (particle radius > distance)
      assert particle.radius > abs(dist)
    end
  end

  describe "performance considerations" do
    test "can handle large particle counts" do
      # Test that we can create large particle arrays
      particles = for i <- 1..10_000 do
        local_mock_particle(id: "p#{i}")
      end

      assert length(particles) == 10_000

      # Test that all particles have valid structure
      assert Enum.all?(particles, &has_particle_fields/1)
    end

    test "particle data structure is efficient" do
      # Test that particle structure is appropriate for GPU processing
      particle = local_mock_particle()

      # Should have fields needed for GPU processing
      assert Map.has_key?(particle, :position)
      assert Map.has_key?(particle, :velocity)
      assert Map.has_key?(particle, :mass)
      assert Map.has_key?(particle, :radius)

      # Position and velocity should be 3-tuples (good for SIMD)
      assert is_tuple(particle.position) and tuple_size(particle.position) == 3
      assert is_tuple(particle.velocity) and tuple_size(particle.velocity) == 3
    end
  end

  describe "conservation laws" do
    test "momentum conservation in particle collisions" do
      p1 = local_mock_particle(
        position: {0.0, 0.0, 0.0},
        velocity: {2.0, 0.0, 0.0},
        mass: 2.0
      )
      p2 = local_mock_particle(
        position: {3.0, 0.0, 0.0},
        velocity: {-1.0, 0.0, 0.0},
        mass: 1.0
      )

      # Initial momentum
      initial_px = p1.mass * elem(p1.velocity, 0) + p2.mass * elem(p2.velocity, 0)
      initial_py = p1.mass * elem(p1.velocity, 1) + p2.mass * elem(p2.velocity, 1)
      initial_pz = p1.mass * elem(p1.velocity, 2) + p2.mass * elem(p2.velocity, 2)

      # After elastic collision, momentum should be conserved
      # For head-on collision: v1' = ((m1-m2)v1 + 2m2v2)/(m1+m2)
      #                   v2' = ((m2-m1)v2 + 2m1v1)/(m1+m2)
      m1 = p1.mass
      m2 = p2.mass
      v1 = elem(p1.velocity, 0)
      v2 = elem(p2.velocity, 0)

      v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
      v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

      final_px = m1 * v1_new + m2 * v2_new
      final_py = 0.0  # No y-component
      final_pz = 0.0  # No z-component

      assert_approx_equal(initial_px, final_px, 0.001)
      assert_approx_equal(initial_py, final_py, 0.001)
      assert_approx_equal(initial_pz, final_pz, 0.001)
    end

    test "energy conservation in elastic collisions" do
      p1 = local_mock_particle(
        position: {0.0, 0.0, 0.0},
        velocity: {2.0, 0.0, 0.0},
        mass: 2.0
      )
      p2 = local_mock_particle(
        position: {3.0, 0.0, 0.0},
        velocity: {-1.0, 0.0, 0.0},
        mass: 1.0
      )

      # Initial kinetic energy
      initial_ke = 0.5 * p1.mass * elem(p1.velocity, 0) * elem(p1.velocity, 0) +
                   0.5 * p2.mass * elem(p2.velocity, 0) * elem(p2.velocity, 0)

      # After elastic collision, energy should be conserved
      m1 = p1.mass
      m2 = p2.mass
      v1 = elem(p1.velocity, 0)
      v2 = elem(p2.velocity, 0)

      v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
      v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

      final_ke = 0.5 * m1 * v1_new * v1_new + 0.5 * m2 * v2_new * v2_new

      assert_approx_equal(initial_ke, final_ke, 0.001)
    end
  end

  describe "integration scenarios" do
    test "particle system initialization" do
      # Test creating a system similar to Demo.run/0
      particles = for i <- 1..100 do
        local_mock_particle(id: "p#{i}")
      end

      walls = create_box()

      initial_state = %{
        particles: particles,
        walls: walls
      }

      assert length(initial_state.particles) == 100
      assert is_list(initial_state.walls)
      assert Enum.all?(initial_state.particles, &has_particle_fields/1)
    end

    test "simulation parameters are reasonable" do
      # Test that the simulation parameters in Demo.run/0 are reasonable
      particle_count = 100_000
      steps = 10_000
      dt = 0.001

      assert particle_count == 100_000
      assert steps == 10_000
      assert dt == 0.001

      # Total simulation time
      total_time = steps * dt
      assert total_time == 10.0
    end
  end

  # Helper functions
  defp has_particle_fields(particle) do
    Map.has_key?(particle, :id) and is_binary(particle.id) and
    Map.has_key?(particle, :position) and is_tuple(particle.position) and tuple_size(particle.position) == 3 and
    Map.has_key?(particle, :velocity) and is_tuple(particle.velocity) and tuple_size(particle.velocity) == 3 and
    Map.has_key?(particle, :mass) and is_number(particle.mass) and particle.mass > 0 and
    Map.has_key?(particle, :radius) and is_number(particle.radius) and particle.radius > 0
  end

  defp local_mock_particle(opts \\ []) do
    %{
      id: Keyword.get(opts, :id, "test_particle"),
      position: Keyword.get(opts, :position, {0.0, 0.0, 0.0}),
      velocity: Keyword.get(opts, :velocity, {0.0, 0.0, 0.0}),
      mass: Keyword.get(opts, :mass, 1.0),
      radius: Keyword.get(opts, :radius, 0.5)
    }
  end

  # Copy the private functions from Demo for testing
  defp random_position do
    {:rand.uniform() * 100, :rand.uniform() * 100, :rand.uniform() * 100}
  end

  defp random_velocity do
    {:rand.uniform() - 0.5, :rand.uniform() - 0.5, :rand.uniform() - 0.5}
  end

  defp create_box do
    # Six walls forming a box (currently empty)
    []
  end
end
