defmodule Examples.GravitySimulationTest do
  use ExUnit.Case
  alias Examples.GravitySimulation
  import Examples.TestHelper

  describe "GravitySimulation.solar_system/0" do
    test "creates a valid solar system configuration" do
      system = GravitySimulation.solar_system()

      # Check structure
      assert is_map(system)
      assert is_list(system.bodies)

      # Check body count
      assert length(system.bodies) == 4

      # Check body properties
      Enum.each(system.bodies, &has_body_fields/1)

      # Check specific bodies
      sun = Enum.find(system.bodies, &(&1.mass == 1000.0))
      assert sun != nil
      assert sun.position == {0.0, 0.0, 0.0}
      assert sun.velocity == {0.0, 0.0, 0.0}

      earth = Enum.find(system.bodies, &(&1.mass == 1.0))
      assert earth != nil
      assert earth.position == {50.0, 0.0, 0.0}
      assert earth.velocity == {0.0, 4.5, 0.0}
    end

    test "solar system has realistic mass distribution" do
      system = GravitySimulation.solar_system()
      masses = Enum.map(system.bodies, & &1.mass)

      # Sun should be most massive
      assert Enum.max(masses) == 1000.0

      # Planets should have decreasing masses
      planet_masses = Enum.reject(masses, &(&1 == 1000.0))
      assert Enum.sort(planet_masses) == [0.5, 1.0, 10.0]
    end
  end

  describe "GravitySimulation.galaxy/1" do
    test "creates a valid galaxy configuration" do
      galaxy = GravitySimulation.galaxy(100)

      # Check structure
      assert is_map(galaxy)
      assert is_list(galaxy.bodies)
      assert length(galaxy.bodies) == 101  # 1 central + 100 orbiting

      # Check central body
      central = hd(galaxy.bodies)
      assert central.mass == 10000.0
      assert central.position == {0.0, 0.0, 0.0}
      assert central.velocity == {0.0, 0.0, 0.0}

      # Check orbiting bodies
      orbiting = tl(galaxy.bodies)
      Enum.each(orbiting, fn body ->
        assert has_body_fields(body)
        assert body.mass >= 0.1
        assert body.mass <= 0.6
        assert body.radius == 0.5
      end)
    end

    test "orbiting bodies have circular orbits" do
      galaxy = GravitySimulation.galaxy(10)
      orbiting = tl(galaxy.bodies)

      Enum.each(orbiting, fn body ->
        {x, y, z} = body.position
        {vx, vy, vz} = body.velocity

        # Check that velocity is perpendicular to position (circular orbit)
        dot_product = x*vx + y*vy + z*vz
        assert_approx_equal(dot_product, 0.0, 0.1)

        # Check orbital speed
        radius = distance(body.position, {0.0, 0.0, 0.0})
        speed = :math.sqrt(vx*vx + vy*vy + vz*vz)
        expected_speed = :math.sqrt(10000.0 / radius)
        assert_approx_equal(speed, expected_speed, 0.1)
      end)
    end
  end

  describe "GravitySimulation.cluster/1" do
    test "creates a valid cluster configuration" do
      cluster = GravitySimulation.cluster(50)

      # Check structure
      assert is_map(cluster)
      assert is_list(cluster.bodies)
      assert length(cluster.bodies) == 50

      # Check body properties
      Enum.each(cluster.bodies, fn body ->
        assert has_body_fields(body)
        assert body.mass >= 0.5
        assert body.mass <= 2.5
        assert body.radius >= 0.5
        assert body.radius <= 1.5
      end)
    end

    test "cluster bodies are randomly distributed" do
      cluster = GravitySimulation.cluster(100)

      # Check position distribution
      positions = Enum.map(cluster.bodies, & &1.position)
      xs = Enum.map(positions, fn {x, _, _} -> x end)
      ys = Enum.map(positions, fn {_, y, _} -> y end)
      zs = Enum.map(positions, fn {_, _, z} -> z end)

      # Should span reasonable range
      assert Enum.min(xs) < -40.0
      assert Enum.max(xs) > 40.0
      assert Enum.min(ys) < -40.0
      assert Enum.max(ys) > 40.0
      assert Enum.min(zs) < -40.0
      assert Enum.max(zs) > 40.0
    end
  end

  describe "GravitySimulation.total_energy/1" do
    test "calculates total energy correctly" do
      system = GravitySimulation.solar_system()
      total_energy = GravitySimulation.total_energy(system)

      assert is_number(total_energy)
      assert total_energy < 0  # Bound system should have negative energy
    end

    test "energy is conserved in simulation" do
      initial_system = GravitySimulation.solar_system()
      initial_energy = GravitySimulation.total_energy(initial_system)

      # Run simulation
      final_system = GravitySimulation.simulate(initial_system, steps: 100, dt: 0.01)
      final_energy = GravitySimulation.total_energy(final_system)

      # Energy should be conserved
      assert_conservation(initial_energy, final_energy, 0.01, "total energy")
    end

    test "energy calculation includes kinetic and potential components" do
      # Simple two-body system
      bodies = [
        mock_body(position: {0.0, 0.0, 0.0}, velocity: {0.0, 0.0, 0.0}, mass: 10.0),
        mock_body(position: {10.0, 0.0, 0.0}, velocity: {0.0, 1.0, 0.0}, mass: 1.0)
      ]

      system = %{bodies: bodies}
      total_energy = GravitySimulation.total_energy(system)

      # Expected: 0.5*1.0*1.0^2 - 10.0*1.0/10.0
      expected_kinetic = 0.5 * 1.0 * 1.0
      expected_potential = -10.0 * 1.0 / 10.0
      expected_total = expected_kinetic + expected_potential

      assert_approx_equal(total_energy, expected_total)
    end
  end

  describe "GravitySimulation.total_momentum/1" do
    test "calculates total momentum correctly" do
      system = GravitySimulation.solar_system()
      total_momentum = GravitySimulation.total_momentum(system)

      assert is_tuple(total_momentum)
      assert tuple_size(total_momentum) == 3

      {px, py, pz} = total_momentum
      assert is_number(px) and is_number(py) and is_number(pz)
    end

    test "momentum is conserved in simulation" do
      initial_system = GravitySimulation.solar_system()
      initial_momentum = GravitySimulation.total_momentum(initial_system)

      # Run simulation
      final_system = GravitySimulation.run_simulation(initial_system, steps: 100, dt: 0.01)
      final_momentum = GravitySimulation.total_momentum(final_system)

      # Momentum should be conserved
      {px1, py1, pz1} = initial_momentum
      {px2, py2, pz2} = final_momentum

      assert_approx_equal(px1, px2, 0.01)
      assert_approx_equal(py1, py2, 0.01)
      assert_approx_equal(pz1, pz2, 0.01)
    end

    test "solar system has approximately zero momentum" do
      system = GravitySimulation.solar_system()
      total_momentum = GravitySimulation.total_momentum(system)

      {px, py, pz} = total_momentum

      # Should be close to zero for stable system
      assert_approx_equal(px, 0.0, 0.1)
      assert_approx_equal(py, 0.0, 0.1)
      assert_approx_equal(pz, 0.0, 0.1)
    end
  end

  describe "GravitySimulation.center_of_mass/1" do
    test "calculates center of mass correctly" do
      system = GravitySimulation.solar_system()
      com = GravitySimulation.center_of_mass(system)

      assert is_tuple(com)
      assert tuple_size(com) == 3

      {x, y, z} = com
      assert is_number(x) and is_number(y) and is_number(z)
    end

    test "center of mass is stable in simulation" do
      initial_system = GravitySimulation.solar_system()
      initial_com = GravitySimulation.center_of_mass(initial_system)

      # Run simulation
      final_system = GravitySimulation.simulate(initial_system, steps: 100, dt: 0.01)
      final_com = GravitySimulation.center_of_mass(final_system)

      # Center of mass should be stable
      distance_moved = distance(initial_com, final_com)
      assert distance_moved < 1.0
    end

    test "center of mass calculation is accurate" do
      # Simple two-body system
      bodies = [
        mock_body(position: {0.0, 0.0, 0.0}, mass: 2.0),
        mock_body(position: {10.0, 0.0, 0.0}, mass: 1.0)
      ]

      system = %{bodies: bodies}
      com = GravitySimulation.center_of_mass(system)

      # Expected: (2.0*{0,0,0} + 1.0*{10,0,0}) / 3.0 = {10/3, 0, 0}
      expected_com = {10.0/3.0, 0.0, 0.0}
      assert_vectors_equal(com, expected_com)
    end
  end

  describe "GravitySimulation.verify_conservation/3" do
    test "verifies conservation laws correctly" do
      initial_system = GravitySimulation.solar_system()
      final_system = GravitySimulation.simulate(initial_system, steps: 100, dt: 0.01)

      conservation = GravitySimulation.verify_conservation(initial_system, final_system, 0.01)

      assert is_map(conservation)
      assert Map.has_key?(conservation, :energy_conserved)
      assert Map.has_key?(conservation, :energy_error)
      assert Map.has_key?(conservation, :momentum_conserved)
      assert Map.has_key?(conservation, :momentum_error)
      assert Map.has_key?(conservation, :center_of_mass_stable)
      assert Map.has_key?(conservation, :com_error)

      # Should pass conservation checks
      assert conservation.energy_conserved == true
      assert conservation.momentum_conserved == true
      assert conservation.center_of_mass_stable == true
    end

    test "detects conservation violations" do
      initial_system = GravitySimulation.solar_system()

      # Artificially change energy to violate conservation
      modified_bodies = Enum.map(initial_system.bodies, fn body ->
        %{body | velocity: {elem(body.velocity, 0) * 2, elem(body.velocity, 1), elem(body.velocity, 2)}}
      end)
      final_system = %{initial_system | bodies: modified_bodies}

      conservation = GravitySimulation.verify_conservation(initial_system, final_system, 0.01)

      # Should detect energy conservation violation
      assert conservation.energy_conserved == false
      assert conservation.energy_error > 0.01
    end
  end

  describe "GravitySimulation.simulate/2" do
    test "can run a basic simulation" do
      initial_system = GravitySimulation.solar_system()

      # Test that simulation runs without errors
      assert_performance(fn ->
        GravitySimulation.simulate(initial_system, steps: 10, dt: 0.01)
      end, 5000)  # 5 second timeout
    end

    test "simulation preserves system structure" do
      initial_system = GravitySimulation.solar_system()
      final_system = GravitySimulation.simulate(initial_system, steps: 10, dt: 0.01)

      # Check that structure is preserved
      assert is_map(final_system)
      assert is_list(final_system.bodies)
      assert length(final_system.bodies) == length(initial_system.bodies)

      # Check that body properties are preserved
      Enum.each(final_system.bodies, &has_body_fields/1)
    end

    test "simulation changes body positions over time" do
      initial_system = GravitySimulation.solar_system()
      final_system = GravitySimulation.simulate(initial_system, steps: 100, dt: 0.01)

      # Positions should change
      initial_positions = Enum.map(initial_system.bodies, & &1.position)
      final_positions = Enum.map(final_system.bodies, & &1.position)

      # At least some positions should change
      positions_changed = Enum.any?(Enum.zip(initial_positions, final_positions), fn {init_pos, final_pos} ->
        distance(init_pos, final_pos) > 0.001
      end)

      assert positions_changed
    end

    test "simulation handles different time steps" do
      system = GravitySimulation.solar_system()

      # Test different time steps
      result1 = GravitySimulation.simulate(system, steps: 10, dt: 0.01)
      result2 = GravitySimulation.simulate(system, steps: 5, dt: 0.02)

      # Both should complete without errors
      assert is_map(result1)
      assert is_map(result2)
      assert length(result1.bodies) == length(result2.bodies)
    end
  end

  describe "edge cases" do
    test "handles empty system" do
      empty_system = %{bodies: []}

      total_energy = GravitySimulation.total_energy(empty_system)
      total_momentum = GravitySimulation.total_momentum(empty_system)
      center_of_mass = GravitySimulation.center_of_mass(empty_system)

      assert total_energy == 0.0
      assert total_momentum == {0.0, 0.0, 0.0}
      assert center_of_mass == {0.0, 0.0, 0.0}
    end

    test "handles single body" do
      single_body = %{bodies: [mock_body(mass: 10.0)]}

      total_energy = GravitySimulation.total_energy(single_body)
      total_momentum = GravitySimulation.total_momentum(single_body)
      center_of_mass = GravitySimulation.center_of_mass(single_body)

      # Single body has only kinetic energy
      assert total_energy >= 0.0
      assert total_momentum == {0.0, 0.0, 0.0}
      assert center_of_mass == hd(single_body.bodies).position
    end
  end

  # Helper functions
  defp has_body_fields(body) do
    is_tuple(body.position) and tuple_size(body.position) == 3 and
    is_tuple(body.velocity) and tuple_size(body.velocity) == 3 and
    is_number(body.mass) and body.mass > 0 and
    is_number(body.radius) and body.radius > 0
  end
end
