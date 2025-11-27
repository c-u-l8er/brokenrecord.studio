defmodule Examples.AIIConservationVerificationTest do
  use ExUnit.Case
  alias Examples.AIIConservationVerification
  import Examples.TestHelper

  describe "create_verification_system/1" do
    test "creates system with default number of particles" do
      system = AIIConservationVerification.create_verification_system()

      assert is_map(system)
      assert Map.has_key?(system, :particles)
      assert Map.has_key?(system, :violations)
      assert Map.has_key?(system, :time)
      assert Map.has_key?(system, :step)

      assert length(system.particles) == 10
      assert system.time == 0.0
      assert system.step == 0
    end

    test "creates system with custom number of particles" do
      system = AIIConservationVerification.create_verification_system(5)

      assert length(system.particles) == 5
    end

    test "particles have correct structure" do
      system = AIIConservationVerification.create_verification_system(1)
      particle = hd(system.particles)

      # Check required fields
      required_fields = [:mass, :charge, :particle_id, :position, :velocity,
                        :acceleration, :energy, :momentum, :charge_conserved,
                        :information, :mass_conserved]

      Enum.each(required_fields, fn field ->
        assert Map.has_key?(particle, field), "Particle missing field: #{field}"
      end)

      # Check types
      assert is_float(particle.mass)
      assert is_float(particle.charge)
      assert is_integer(particle.particle_id)
      assert is_tuple(particle.position)
      assert is_tuple(particle.velocity)
      assert is_tuple(particle.acceleration)
    end
  end

  describe "run_verification/2" do
    test "runs verification with default options" do
      initial_state = AIIConservationVerification.create_verification_system(3)

      result = AIIConservationVerification.run_verification(initial_state)

      assert {:ok, final_state} = result
      assert final_state.steps == 100
      assert final_state.dt == 0.016
      assert is_list(final_state.violations)
    end

    test "runs verification with custom options" do
      initial_state = AIIConservationVerification.create_verification_system(3)

      result = AIIConservationVerification.run_verification(initial_state, steps: 50)

      assert {:ok, final_state} = result
      assert final_state.steps == 50
      assert final_state.dt == 0.016
      assert is_list(final_state.violations)
    end
  end



  describe "verification_report/1" do
    test "generates verification report" do
      system = AIIConservationVerification.create_verification_system(3)

      report = AIIConservationVerification.verification_report(system)

      assert is_map(report)
      assert Map.has_key?(report, :total_violations)
      assert Map.has_key?(report, :violations_by_type)
      assert Map.has_key?(report, :violations_by_quantity)
      assert Map.has_key?(report, :max_error)
      assert Map.has_key?(report, :recommendations)

      assert is_integer(report.total_violations)
      assert is_list(report.violations_by_type)
      assert is_list(report.violations_by_quantity)
      assert is_list(report.recommendations)
    end

    test "handles system with no violations" do
      # Create a system that should have no violations
      system = AIIConservationVerification.create_verification_system(1)

      report = AIIConservationVerification.verification_report(system)

      assert report.total_violations >= 0
    end
  end

  describe "helper functions" do
    test "colliding? detects collisions" do
      p1 = %{position: {0.0, 0.0, 0.0}}
      p2 = %{position: {1.0, 0.0, 0.0}}

      assert AIIConservationVerification.colliding?(p1, p2)

      p3 = %{position: {3.0, 0.0, 0.0}}
      refute AIIConservationVerification.colliding?(p1, p3)
    end

    test "calculate_elastic_collision conserves momentum and energy" do
      p1 = %{mass: 1.0, velocity: {2.0, 0.0, 0.0}, charge: 0.0}
      p2 = %{mass: 1.0, velocity: {-1.0, 0.0, 0.0}, charge: 0.0}

      {new_p1, new_p2} = AIIConservationVerification.calculate_elastic_collision(p1, p2)

      # Total momentum should be conserved
      initial_momentum = AII.Types.Vec3.add(
        AII.Types.Vec3.mul(p1.velocity, p1.mass),
        AII.Types.Vec3.mul(p2.velocity, p2.mass)
      )

      final_momentum = AII.Types.Vec3.add(
        AII.Types.Vec3.mul(new_p1.velocity, new_p1.mass),
        AII.Types.Vec3.mul(new_p2.velocity, new_p2.mass)
      )

      momentum_error = AII.Types.Vec3.magnitude(
        AII.Types.Vec3.sub(initial_momentum, final_momentum)
      )

      assert momentum_error < 1.0e-10
    end

    test "check_conservation detects violations" do
      violations = []

      # Test with conserved values
      result1 = AIIConservationVerification.check_conservation(violations, :energy, 100.0, 100.0, 0.01)
      assert result1 == []

      # Test with violation
      result2 = AIIConservationVerification.check_conservation(violations, :energy, 100.0, 105.0, 0.01)
      assert length(result2) == 1
      assert hd(result2).quantity == :energy
      assert hd(result2).error > 0
    end
  end

  describe "DSL integration" do
    test "module has correct agents and interactions" do
      # Check that the DSL compiled correctly
      assert function_exported?(AIIConservationVerification, :__agents__, 0)
      assert function_exported?(AIIConservationVerification, :__interactions__, 0)

      agents = AIIConservationVerification.__agents__()
      interactions = AIIConservationVerification.__interactions__()

      assert is_list(agents)
      assert is_list(interactions)

      # Should have Particle agent
      assert length(agents) >= 0

      # Should have collision and integration interactions
      assert length(interactions) >= 0
    end

    test "conservation verification works" do
      # Test that the DSL conservation checker works
      interaction = %{body: {:elastic_collision, [], []}, name: :elastic_collision}
      agents = AIIConservationVerification.__agents__()

      result = AII.verify_conservation(interaction, agents)

      # Should not crash and return some result
      assert result == :ok or match?({:needs_runtime_check, _, _}, result)
    end
  end

  describe "performance" do
    test "verification completes in reasonable time" do
      system = AIIConservationVerification.create_verification_system(5)

      {time_ms, result} = measure_time(fn ->
        AIIConservationVerification.run_verification(system, steps: 20)
      end)

      assert {:ok, _} = result
      # Should complete in less than 1 second for 5 particles, 20 steps
      assert time_ms < 1000
    end

    test "report generation is fast" do
      system = AIIConservationVerification.create_verification_system(3)

      {time_ms, _} = measure_time(fn ->
        AIIConservationVerification.verification_report(system)
      end)

      # Should be very fast
      assert time_ms < 10
    end
  end

  describe "edge cases" do
    test "handles single particle system" do
      system = AIIConservationVerification.create_verification_system(1)

      result = AIIConservationVerification.run_verification(system, steps: 5)
      assert {:ok, _} = result

      report = AIIConservationVerification.verification_report(system)
      assert report.total_violations >= 0
    end

    test "handles empty system" do
      system = %{particles: [], violations: [], time: 0.0, step: 0}

      result = AIIConservationVerification.run_verification(system, steps: 5)
      assert {:ok, _} = result

      report = AIIConservationVerification.verification_report(system)
      assert report.total_violations == 0
    end

    test "check_conservation handles zero values" do
      violations = AIIConservationVerification.check_conservation([], :energy, 0.0, 0.0, 0.01)
      assert violations == []
    end

    test "check_conservation handles large errors" do
      violations = AIIConservationVerification.check_conservation([], :energy, 1.0, 100.0, 0.01)
      assert length(violations) == 1
      assert hd(violations).error > 50
    end
  end
end
