defmodule Examples.GravityTest do
  use ExUnit.Case
  import Examples.TestHelper
  alias Examples.Gravity

  describe "gravity simulation" do
    test "creates simple system with one particle" do
      particles = Gravity.create_simple_system()
      assert length(particles) == 1

      particle = hd(particles)
      assert particle.mass == 1.0
      assert particle.position == {0.0, 10.0, 0.0}
      assert particle.velocity == {0.0, 0.0, 0.0}
      assert particle.kinetic_energy.value == 0.0
      assert particle.momentum.value == {0.0, 0.0, 0.0}
    end

    test "applies gravity to particles" do
      particles = Gravity.create_simple_system()
      updated = Gravity.apply_gravity(particles, 1.0)

      particle = hd(updated)
      # Velocity should be {0, -9.81, 0} after gravity
      assert particle.velocity == {0.0, -9.81, 0.0}
      assert particle.position == {0.0, 10.0, 0.0}  # Position unchanged
    end

    test "integrates motion correctly" do
      particles = Gravity.create_simple_system()
      with_gravity = Gravity.apply_gravity(particles, 1.0)
      integrated = Gravity.integrate_motion(with_gravity, 1.0)

      particle = hd(integrated)
      # Position should be {0, 10 + (-9.81)*1, 0} = {0, 0.19, 0}
      assert_approx_equal(elem(particle.position, 0), 0.0)
      assert_approx_equal(elem(particle.position, 1), 10.0 - 9.81)
      assert_approx_equal(elem(particle.position, 2), 0.0)

      # Velocity should still be {0, -9.81, 0}
      assert particle.velocity == {0.0, -9.81, 0.0}

      # Kinetic energy should be 0.5 * 1 * (9.81)^2
      expected_ke = 0.5 * 1.0 * 9.81 * 9.81
      assert_approx_equal(particle.kinetic_energy.value, expected_ke)

      # Momentum should be mass * velocity = {0, -9.81, 0}
      assert particle.momentum.value == {0.0, -9.81, 0.0}
    end

    test "simulates one step correctly" do
      particles = Gravity.create_simple_system()
      after_step = Gravity.simulate_step(particles, 1.0)

      particle = hd(after_step)
      # Should have applied gravity and integrated motion
      assert_approx_equal(elem(particle.position, 0), 0.0)
      assert_approx_equal(elem(particle.position, 1), 10.0 - 9.81)
      assert_approx_equal(elem(particle.position, 2), 0.0)
      assert particle.velocity == {0.0, -9.81, 0.0}
    end

    test "energy conservation (approximately)" do
      particles = Gravity.create_simple_system()
      initial_energy = Gravity.total_energy(particles)

      # Run a few steps
      after_steps = Enum.reduce(1..5, particles, fn _, acc ->
        Gravity.simulate_step(acc, 1.0)
      end)

      final_energy = Gravity.total_energy(after_steps)

      # Total energy (kinetic + potential) should be approximately conserved
      # (allowing for numerical error in Euler integration)
      assert_conservation(initial_energy, final_energy, 100.0, "energy")
    end

    test "momentum changes due to gravity" do
      particles = Gravity.create_simple_system()
      _initial_momentum = Gravity.total_momentum(particles)

      # Run 5 steps
      after_steps = Enum.reduce(1..5, particles, fn _, acc ->
        Gravity.simulate_step(acc, 1.0)
      end)

      final_momentum = Gravity.total_momentum(after_steps)

      # Momentum should change due to gravity force
      # After 5 steps with Euler integration, velocity accumulates: 5 * (-9.81) = -49.05
      # Momentum = mass * velocity = 1.0 * {0, -49.05, 0} = {0, -49.05, 0}
      expected_momentum = {0.0, -49.05, 0.0}

      assert_vectors_equal(final_momentum, expected_momentum, 0.01)
    end

    test "particle falls downward over time" do
      particles = Gravity.create_simple_system()
      initial_y = elem(hd(particles).position, 1)

      # Run 10 steps
      final_particles = Enum.reduce(1..10, particles, fn _, acc ->
        Gravity.simulate_step(acc, 1.0)
      end)

      final_y = elem(hd(final_particles).position, 1)

      # Should be lower than initial (falling)
      assert final_y < initial_y

      # With proper Euler integration: position = 10 + sum of (velocity * dt)
      # But since velocity increases each step, it's more complex
      # For now, just check it's falling
      assert final_y < -50.0  # Should be well below starting point
    end

    test "run_simulation completes without error" do
      # Capture IO to avoid cluttering test output
      captured = ExUnit.CaptureIO.capture_io(fn ->
        result = Gravity.run_simulation(3)
        assert result == :ok
      end)

      # Should have output simulation steps
      assert String.contains?(captured, "Initial state:")
      assert String.contains?(captured, "Step 1:")
      assert String.contains?(captured, "Step 2:")
      assert String.contains?(captured, "Step 3:")
      assert String.contains?(captured, "Conservation Check:")
    end

    test "performance is reasonable" do
      particles = Gravity.create_simple_system()

      # Measure time for 100 simulation steps
      {_time_ms, _} = measure_time(fn ->
        Enum.reduce(1..100, particles, fn _, acc ->
          Gravity.simulate_step(acc, 1.0)
        end)
      end)

      # Should complete in reasonable time (less than 1 second)
      assert_performance(fn ->
        Enum.reduce(1..100, particles, fn _, acc ->
          Gravity.simulate_step(acc, 1.0)
        end)
      end, 1000)
    end
  end

  describe "derived quantities" do
    test "potential energy calculation" do
      particle = %{
        mass: 2.0,
        position: {0.0, 5.0, 0.0},
        velocity: {0.0, 0.0, 0.0},
        kinetic_energy: AII.Types.Conserved.new(0.0, "test"),
        momentum: AII.Types.Conserved.new({0.0, 0.0, 0.0}, "test")
      }

      # Potential energy = -mass * g * height = -2 * 9.81 * 5
      expected_pe = -2.0 * 9.81 * 5.0
      actual_pe = Gravity.total_energy([particle]) - 0  # KE is 0

      assert_approx_equal(actual_pe, expected_pe)
    end

    test "kinetic energy updates with velocity" do
      # Create particle with velocity
      particle = %{
        mass: 1.0,
        position: {0.0, 0.0, 0.0},
        velocity: {3.0, 4.0, 0.0},  # Speed = 5
        kinetic_energy: AII.Types.Conserved.new(0.0, "test"),
        momentum: AII.Types.Conserved.new({0.0, 0.0, 0.0}, "test")
      }

      # Update kinetic energy
      updated = Gravity.integrate_motion([particle], 1.0)
      updated_particle = hd(updated)

      # KE = 0.5 * mass * speed^2 = 0.5 * 1 * 25 = 12.5
      expected_ke = 0.5 * 1.0 * (3.0*3.0 + 4.0*4.0)
      assert_approx_equal(updated_particle.kinetic_energy.value, expected_ke)
    end
  end
end
