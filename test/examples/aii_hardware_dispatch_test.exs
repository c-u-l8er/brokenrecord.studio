defmodule Examples.AIIHardwareDispatchTest do
  use ExUnit.Case
  alias Examples.AIIHardwareDispatch
  import Examples.TestHelper

  describe "create_hardware_demo/1" do
    test "creates demo with default number of particles" do
      system = AIIHardwareDispatch.create_hardware_demo()

      assert is_map(system)
      assert Map.has_key?(system, :particles)
      assert Map.has_key?(system, :spatial_grid)
      assert Map.has_key?(system, :time)
      assert Map.has_key?(system, :step)

      assert length(system.particles) == 1000
      assert system.time == 0.0
      assert system.step == 0
    end

    test "creates demo with custom number of particles" do
      system = AIIHardwareDispatch.create_hardware_demo(500)

      assert length(system.particles) == 500
    end

    test "particles have correct structure" do
      system = AIIHardwareDispatch.create_hardware_demo(1)
      particle = hd(system.particles)

      # Check required fields
      required_fields = [:mass, :radius, :particle_type, :position, :velocity,
                        :acceleration, :color, :energy, :momentum, :information]

      Enum.each(required_fields, fn field ->
        assert Map.has_key?(particle, field), "Particle missing field: #{field}"
      end)

      # Check types
      assert is_float(particle.mass)
      assert is_float(particle.radius)
      assert particle.particle_type in [:matter, :antimatter, :dark_matter]
      assert is_tuple(particle.position)
      assert is_tuple(particle.velocity)
      assert is_tuple(particle.acceleration)
      assert is_tuple(particle.color)
    end

    test "spatial grid is properly initialized" do
      system = AIIHardwareDispatch.create_hardware_demo(10)

      assert is_map(system.spatial_grid)
      assert Map.has_key?(system.spatial_grid, :grid_size)
      assert Map.has_key?(system.spatial_grid, :cell_size)
      assert Map.has_key?(system.spatial_grid, :cells)
      assert Map.has_key?(system.spatial_grid, :particle_count)

      assert system.spatial_grid.particle_count == 10
    end
  end

  describe "run_simulation/2" do
    test "runs simulation with default options" do
      initial_state = AIIHardwareDispatch.create_hardware_demo(50)

      result = AIIHardwareDispatch.run_simulation(initial_state)

      assert is_map(result)
      assert result.steps == 1000
      assert result.dt == 0.016
    end

    test "runs simulation with custom options" do
      initial_state = AIIHardwareDispatch.create_hardware_demo(25)

      result = AIIHardwareDispatch.run_simulation(initial_state, steps: 100, dt: 0.01)

      assert is_map(result)
      assert result.steps == 100
      assert result.dt == 0.01
    end
  end

  describe "hardware_stats/1" do
    test "calculates hardware statistics" do
      system = AIIHardwareDispatch.create_hardware_demo(20)

      stats = AIIHardwareDispatch.hardware_stats(system)

      assert is_map(stats)
      assert Map.has_key?(stats, :total_particles)
      assert Map.has_key?(stats, :total_energy)
      assert Map.has_key?(stats, :total_momentum)
      assert Map.has_key?(stats, :total_information)
      assert Map.has_key?(stats, :spatial_grid_efficiency)
      assert Map.has_key?(stats, :spatial_grid_efficiency)

      assert stats.total_particles == 20
      assert is_float(stats.total_energy)
      assert is_tuple(stats.total_momentum)
      assert is_float(stats.total_information)
      assert is_float(stats.spatial_grid_efficiency)
      assert is_map(stats.hardware_utilization)
    end

    test "handles empty system" do
      empty_system = %{particles: [], spatial_grid: %{particle_count: 0}, time: 0.0, step: 0}

      stats = AIIHardwareDispatch.hardware_stats(empty_system)

      assert stats.total_particles == 0
      assert stats.total_energy == 0.0
      assert stats.total_information == 0.0
    end
  end

  describe "detect_hardware/0" do
    test "detects available hardware" do
      hardware = AIIHardwareDispatch.detect_hardware()

      assert is_map(hardware)
      assert Map.has_key?(hardware, :rt_cores_available)
      assert Map.has_key?(hardware, :tensor_cores_available)
      assert Map.has_key?(hardware, :npu_available)
      assert Map.has_key?(hardware, :cuda_available)
      assert Map.has_key?(hardware, :gpu_available)
      assert Map.has_key?(hardware, :parallel_available)
      assert Map.has_key?(hardware, :simd_available)

      # All should be boolean except core_count
      Enum.each(hardware, fn {key, value} ->
        if key != :core_count, do: assert is_boolean(value)
      end)
    end
  end

  describe "optimization_recommendations/0" do
    test "provides optimization recommendations" do
      recommendations = AIIHardwareDispatch.optimization_recommendations()

      assert is_map(recommendations)
      assert Map.has_key?(recommendations, :primary_accelerator)
      assert Map.has_key?(recommendations, :secondary_accelerator)
      assert Map.has_key?(recommendations, :fallback)
      assert Map.has_key?(recommendations, :optimization_tips)

      # optimization_tips should be a list
      assert is_list(recommendations.optimization_tips)
      assert length(recommendations.optimization_tips) > 0
    end
  end

  describe "DSL integration" do
    test "module has correct agents and interactions" do
      # Check that the DSL compiled correctly
      assert function_exported?(AIIHardwareDispatch, :__agents__, 0)
      assert function_exported?(AIIHardwareDispatch, :__interactions__, 0)

      agents = AIIHardwareDispatch.__agents__()
      interactions = AIIHardwareDispatch.__interactions__()

      assert is_list(agents)
      assert is_list(interactions)

      # Should have Particle and SpatialGrid agents
      assert length(agents) >= 2

      # Should have collision, matrix, npu, and diffusion interactions
      assert length(interactions) >= 4
    end

    test "conservation verification works" do
      # Test that the DSL conservation checker works
      interaction = %{body: {:rt_collision_detection, [], []}, name: :rt_collision_detection}
      agents = AIIHardwareDispatch.__agents__()

      result = AII.verify_conservation(interaction, agents)

      # Should not crash and return some result
      assert result == :ok or match?({:needs_runtime_check, _, _}, result)
    end
  end

  describe "performance" do
    test "simulation completes in reasonable time" do
      system = AIIHardwareDispatch.create_hardware_demo(50)

      {time_ms, result} = measure_time(fn ->
        AIIHardwareDispatch.run_simulation(system, steps: 20)
      end)

      assert is_map(result)
      # Should complete in reasonable time for 50 particles, 20 steps
      assert time_ms < 5000
    end

    test "hardware detection is fast" do
      {time_ms, _} = measure_time(fn ->
        AIIHardwareDispatch.detect_hardware()
      end)

      # Should be very fast
      assert time_ms < 10
    end

    test "recommendations generation is fast" do
      {time_ms, _} = measure_time(fn ->
        AIIHardwareDispatch.optimization_recommendations()
      end)

      # Should be very fast
      assert time_ms < 10
    end
  end

  describe "edge cases" do
    test "handles single particle system" do
      system = AIIHardwareDispatch.create_hardware_demo(1)

      result = AIIHardwareDispatch.run_simulation(system, steps: 5)
      assert is_map(result)

      stats = AIIHardwareDispatch.hardware_stats(system)
      assert stats.total_particles == 1
    end

    test "handles empty system" do
      system = %{particles: [], spatial_grid: %{particle_count: 0, cells: %{}}, time: 0.0, step: 0}

      result = AIIHardwareDispatch.run_simulation(system, steps: 5)
      assert is_map(result)

      stats = AIIHardwareDispatch.hardware_stats(system)
      assert stats.total_particles == 0
    end

    test "detect_hardware handles all platforms" do
      hardware = AIIHardwareDispatch.detect_hardware()

      # Should always return a complete map
      required_keys = [:rt_cores_available, :tensor_cores_available, :npu_available,
                      :cuda_available, :gpu_available, :parallel_available, :simd_available]

      Enum.each(required_keys, fn key ->
        assert Map.has_key?(hardware, key)
      end)
    end

    test "optimization_recommendations adapts to hardware" do
      recommendations = AIIHardwareDispatch.optimization_recommendations()

      # Should provide useful recommendations for unknown platform
      assert is_map(recommendations)
      assert recommendations.primary_accelerator == :cpu
      assert recommendations.secondary_accelerator == :parallel
      assert :simd in recommendations.fallback
      assert :cpu in recommendations.fallback
      assert is_list(recommendations.optimization_tips)
    end
  end

  describe "spatial grid functionality" do
    test "build_spatial_hash creates proper grid" do
      particles = [
        %{position: {0.0, 0.0, 0.0}},
        %{position: {1.0, 1.0, 1.0}},
        %{position: {10.0, 10.0, 10.0}}
      ]

      grid = %{grid_size: 10, cell_size: 2.0}
      hash = AIIHardwareDispatch.build_spatial_hash(particles, grid)

      assert is_map(hash)
      assert map_size(hash) > 0

      # Should have particles in different cells
      cell_keys = Map.keys(hash)
      assert length(cell_keys) >= 2
    end

    test "query_nearby finds particles in radius" do
      position = {5.0, 5.0, 5.0}
      radius = 3.0
      grid = %{
        cell_size: 10.0,
        cells: %{
          {2, 2, 2} => [%{position: {4.5, 4.5, 4.5}}],
          {3, 3, 3} => [%{position: {6.0, 6.0, 6.0}}],
          {10, 10, 10} => [%{position: {20.0, 20.0, 20.0}}]
        }
      }

      nearby = AIIHardwareDispatch.query_nearby(position, radius, grid)

      assert is_list(nearby)
      assert length(nearby) >= 1
      # Should find the close particles but not the far one
    end
  end

  describe "helper functions" do
    test "get_cell_key calculates correct cell" do
      position = {5.5, 3.2, 7.8}
      cell_size = 2.0

      cell_key = AIIHardwareDispatch.get_cell_key(position, cell_size)

      assert is_tuple(cell_key)
      assert tuple_size(cell_key) == 3

      # 5.5 / 2.0 = 2.75 -> floor to 2
      # 3.2 / 2.0 = 1.6 -> floor to 1
      # 7.8 / 2.0 = 3.9 -> floor to 3
      assert cell_key == {2, 1, 3}
    end

    test "get_nearby_cell_keys returns adjacent cells" do
      position = {5.0, 5.0, 5.0}
      radius = 2.0
      cell_size = 2.0

      nearby_cells = AIIHardwareDispatch.get_nearby_cell_keys(position, radius, cell_size)

      assert is_list(nearby_cells)
      assert length(nearby_cells) > 1

      # Should include the center cell
      assert {2, 2, 2} in nearby_cells
    end
  end
end
