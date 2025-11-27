

defmodule Examples.AIIParticlePhysicsTest do
  use ExUnit.Case
  alias Examples.AIIParticlePhysics
  import Examples.TestHelper

  describe "create_particle_system/1" do
    test "creates system with default number of particles" do
      system = AIIParticlePhysics.create_particle_system()

      assert is_map(system)
      assert Map.has_key?(system, :particles)
      assert Map.has_key?(system, :fields)
      assert Map.has_key?(system, :time)
      assert Map.has_key?(system, :step)

      assert length(system.particles) == 100
      assert length(system.fields) == 2
      assert system.time == 0.0
      assert system.step == 0
    end

    test "creates system with custom number of particles" do
      system = AIIParticlePhysics.create_particle_system(50)

      assert length(system.particles) == 50
    end

    test "particles have correct structure" do
      system = AIIParticlePhysics.create_particle_system(1)
      particle = hd(system.particles)

      # Check required fields
      required_fields = [:mass, :charge, :particle_id, :position, :velocity,
                        :acceleration, :energy, :momentum, :information]

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

    test "fields have correct structure" do
      system = AIIParticlePhysics.create_particle_system(1)
      field = hd(system.fields)

      required_fields = [:field_type, :strength, :direction, :active, :energy_source]

      Enum.each(required_fields, fn req_field ->
        assert Map.has_key?(field, req_field), "Field missing #{req_field}"
      end)

      assert field.field_type in [:gravity, :electric, :magnetic]
      assert is_float(field.strength)
      assert is_tuple(field.direction)
      assert is_boolean(field.active)
    end
  end

  describe "run_simulation/2" do
    test "runs simulation with default options" do
      initial_state = AIIParticlePhysics.create_particle_system(10)

      result = AIIParticlePhysics.run_simulation(initial_state)

      assert {:ok, final_state} = result
      assert final_state.steps == 1000
      assert final_state.dt == 0.016
      assert is_list(final_state.results)
    end

    test "runs simulation with custom options" do
      initial_state = AIIParticlePhysics.create_particle_system(5)

      result = AIIParticlePhysics.run_simulation(initial_state, steps: 100, dt: 0.01)

      assert {:ok, final_state} = result
      assert final_state.steps == 100
      assert final_state.dt == 0.01
    end
  end

  describe "verify_conservation/3" do
    test "verifies conservation for simple system" do
      initial_state = AIIParticlePhysics.create_particle_system(2)

      # Create a mock final state
      final_state = %{initial_state |
        particles: Enum.map(initial_state.particles, fn p ->
          # Simulate some changes but maintain conservation
          %{p |
            position: AII.Types.Vec3.add(p.position, {1.0, 0.0, 0.0}),
            energy: AII.Types.Conserved.new(p.energy.value, :updated)
          }
        end)
      }

      result = AIIParticlePhysics.verify_conservation(initial_state, final_state)

      assert is_map(result)
      assert Map.has_key?(result, :energy_conserved)
      assert Map.has_key?(result, :momentum_conserved)
      assert Map.has_key?(result, :information_conserved)
      assert Map.has_key?(result, :total_particles)
    end

    test "detects conservation violations" do
      initial_state = AIIParticlePhysics.create_particle_system(1)

      # Create final state with energy violation
      final_state = %{initial_state |
        particles: Enum.map(initial_state.particles, fn p ->
          %{p | energy: AII.Types.Conserved.new(p.energy.value + 10.0, :violated)}
        end)
      }

      result = AIIParticlePhysics.verify_conservation(initial_state, final_state)

      refute result.energy_conserved
      assert abs(result.energy_error) > 5.0
    end
  end

  describe "system_stats/1" do
    test "calculates system statistics" do
      system = AIIParticlePhysics.create_particle_system(3)

      stats = AIIParticlePhysics.system_stats(system)

      assert is_map(stats)
      assert stats.total_particles == 3
      assert is_float(stats.total_mass)
      assert is_float(stats.total_charge)
      assert is_float(stats.average_kinetic_energy)
      assert is_tuple(stats.total_momentum)
      assert is_float(stats.total_energy)
      assert is_float(stats.total_information)
      assert stats.simulation_time == 0.0
      assert stats.simulation_step == 0
    end

    test "handles empty system" do
      empty_system = %{particles: [], fields: [], time: 0.0, step: 0}

      stats = AIIParticlePhysics.system_stats(empty_system)

      assert stats.total_particles == 0
      assert stats.total_mass == 0.0
      assert stats.total_charge == 0.0
    end
  end

  describe "helper functions" do
    test "colliding? detects collisions" do
      p1 = %{position: {0.0, 0.0, 0.0}}
      p2 = %{position: {1.0, 0.0, 0.0}}

      assert AIIParticlePhysics.colliding?(p1, p2)  # Distance = 1.0 < 2.0

      p3 = %{position: {3.0, 0.0, 0.0}}
      refute AIIParticlePhysics.colliding?(p1, p3)  # Distance = 3.0 > 2.0
    end

    test "exchange_momentum conserves momentum" do
      p1 = %{mass: 1.0, velocity: {2.0, 0.0, 0.0}}
      p2 = %{mass: 1.0, velocity: {-1.0, 0.0, 0.0}}

      {new_p1, new_p2} = AIIParticlePhysics.exchange_momentum(p1, p2)

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

    test "calculate_field_force computes correct forces" do
      particle = %{mass: 1.0, charge: 1.0, velocity: {1.0, 0.0, 0.0}}

      # Gravity field
      gravity_field = %{field_type: :gravity, strength: 9.81, direction: {0.0, -1.0, 0.0}}
      gravity_force = AIIParticlePhysics.calculate_field_force(particle, gravity_field)
      expected_gravity = {0.0, -9.81, 0.0}  # mass * strength * direction
      assert_vectors_equal(gravity_force, expected_gravity, 0.01)

      # Electric field
      electric_field = %{field_type: :electric, strength: 5.0, direction: {1.0, 0.0, 0.0}}
      electric_force = AIIParticlePhysics.calculate_field_force(particle, electric_field)
      expected_electric = {5.0, 0.0, 0.0}  # charge * strength * direction
      assert_vectors_equal(electric_force, expected_electric, 0.01)

      # Magnetic field (Lorentz force)
      magnetic_field = %{field_type: :magnetic, strength: 2.0, direction: {0.0, 1.0, 0.0}}
      magnetic_force = AIIParticlePhysics.calculate_field_force(particle, magnetic_field)
      # F = q(v × B) = 1.0 * ({1,0,0} × {0,1,0}) * 2.0 = {0,0,1} * 2.0 = {0,0,2}
      expected_magnetic = {0.0, 0.0, 2.0}
      assert_vectors_equal(magnetic_force, expected_magnetic, 0.01)
    end
  end

  describe "DSL integration" do
    test "module has correct agents and interactions" do
      # Check that the DSL compiled correctly
      assert function_exported?(AIIParticlePhysics, :__agents__, 0)
      assert function_exported?(AIIParticlePhysics, :__interactions__, 0)

      agents = AIIParticlePhysics.__agents__()
      interactions = AIIParticlePhysics.__interactions__()

      assert is_list(agents)
      assert is_list(interactions)

      # Should have Particle and Field agents
      assert length(agents) >= 2

      # Should have gravity, collision, field, and integration interactions
      assert length(interactions) >= 4
    end

    test "conservation verification works" do
      # Test that the DSL conservation checker works
      interaction = %{body: {:apply_gravity, [], []}, name: :apply_gravity}
      agents = AIIParticlePhysics.__agents__()

      result = AII.verify_conservation(interaction, agents)

      # Should not crash and return some result
      assert result == :ok or match?({:needs_runtime_check, _, _}, result)
    end
  end

  describe "performance" do
    test "simulation completes in reasonable time" do
      system = AIIParticlePhysics.create_particle_system(10)

      {time_ms, result} = measure_time(fn ->
        AIIParticlePhysics.run_simulation(system, steps: 50)
      end)

      assert {:ok, _} = result
      # Should complete in less than 1 second for 10 particles, 50 steps
      assert time_ms < 1000
    end

    test "conservation verification is fast" do
      initial = AIIParticlePhysics.create_particle_system(5)
      final = %{initial | particles: initial.particles}  # Same particles

      {time_ms, _} = measure_time(fn ->
        AIIParticlePhysics.verify_conservation(initial, final)
      end)

      # Should be very fast
      assert time_ms < 10
    end
  end

  describe "edge cases" do
    test "handles single particle system" do
      system = AIIParticlePhysics.create_particle_system(1)

      result = AIIParticlePhysics.run_simulation(system, steps: 10)
      assert {:ok, _} = result

      stats = AIIParticlePhysics.system_stats(system)
      assert stats.total_particles == 1
    end

    test "handles zero particles" do
      system = %{particles: [], fields: [], time: 0.0, step: 0}

      result = AIIParticlePhysics.run_simulation(system, steps: 5)
      assert {:ok, _} = result

      stats = AIIParticlePhysics.system_stats(system)
      assert stats.total_particles == 0
    end

    test "conservation check handles empty systems" do
      initial = %{particles: [], fields: [], time: 0.0, step: 0}
      final = initial

      result = AIIParticlePhysics.verify_conservation(initial, final)

      assert result.energy_conserved
      assert result.momentum_conserved
      assert result.information_conserved
      assert result.total_particles == 0
    end
  end
end
