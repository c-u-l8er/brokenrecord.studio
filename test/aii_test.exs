defmodule AIITest do
  use ExUnit.Case
  alias AII

  describe "module access" do
    test "provides access to submodules" do
      # Test that module access functions work
      assert AII.types() == AII.Types
      assert AII.dsl() == AII.DSL
      assert AII.hardware() == AII.HardwareDispatcher
      assert AII.codegen() == AII.Codegen
      assert AII.conservation() == AII.ConservationChecker
      assert AII.nif() == AII.NIF
    end
  end

  describe "system_info/0" do
    test "returns system information" do
      info = AII.system_info()

      assert is_map(info)
      assert info.version == "0.1.0"
      assert is_list(info.hardware)
      assert :cpu in info.hardware

      assert is_map(info.performance_hints)
      assert is_number(info.performance_hints.cpu)
      assert info.performance_hints.cpu == 1.0

      assert is_map(info.memory_hints)
      assert is_map(info.efficiency_hints)
    end
  end

  describe "create_particle/1" do
    test "creates particle with defaults" do
      particle = AII.create_particle()

      assert particle.mass == 1.0
      assert particle.position == {0.0, 0.0, 0.0}
      assert particle.velocity == {0.0, 0.0, 0.0}
      assert particle.energy == 0.0
      assert particle.momentum == {0.0, 0.0, 0.0}
    end

    test "creates particle with custom values" do
      particle = AII.create_particle(
        mass: 2.5,
        position: {1.0, 2.0, 3.0},
        velocity: {0.1, 0.2, 0.3},
        energy: 100.0,
        momentum: {10.0, 20.0, 30.0}
      )

      assert particle.mass == 2.5
      assert particle.position == {1.0, 2.0, 3.0}
      assert particle.velocity == {0.1, 0.2, 0.3}
      assert particle.energy == 100.0
      assert particle.momentum == {10.0, 20.0, 30.0}
    end
  end

  describe "available_hardware/0" do
    test "returns list of available hardware" do
      hardware = AII.available_hardware()
      assert is_list(hardware)
      assert :cpu in hardware
    end
  end

  describe "performance_hint/1" do
    test "returns performance hints for different hardware" do
      assert AII.performance_hint(:cpu) == 1.0
      assert AII.performance_hint(:gpu) == 50.0
      assert AII.performance_hint(:cuda_cores) == 100.0
      assert AII.performance_hint(:tensor_cores) == 200.0
      assert AII.performance_hint(:rt_cores) == 150.0
      assert AII.performance_hint(:npu) == 300.0
    end

    test "parallel hint scales with CPU cores" do
      expected = System.schedulers_online() * 0.8
      assert AII.performance_hint(:parallel) == expected
    end
  end

  describe "memory_hint/1" do
    test "returns memory hints for different hardware" do
      assert AII.memory_hint(:cpu) == 1.0
      assert AII.memory_hint(:gpu) == 2.0
      assert AII.memory_hint(:tensor_cores) == 3.0
      assert AII.memory_hint(:rt_cores) == 4.0
      assert AII.memory_hint(:npu) == 1.5
    end
  end

  describe "efficiency_hint/1" do
    test "returns efficiency hints for different hardware" do
      assert AII.efficiency_hint(:cpu) == 1.0
      assert AII.efficiency_hint(:gpu) == 0.5
      assert AII.efficiency_hint(:npu) == 2.0
    end
  end

  describe "run_simulation/2" do
    test "fails for invalid system module" do
      assert_raise UndefinedFunctionError, fn ->
        AII.run_simulation(InvalidModule)
      end
    end

    test "runs simulation with mock system" do
      # Create a mock system module
      defmodule MockSystem do
        def __agents__, do: [%{conserves: [:energy]}]
        def __interactions__, do: [%{body: {:simple, [], []}}]
      end

      # Should return simulation results
      result = AII.run_simulation(MockSystem, steps: 1, particles: [])
      assert {:ok, results} = result
      assert results.steps == 1
      assert results.conservation_verified == true
    end
  end

  describe "benchmark/2" do
    test "benchmarks a simulation" do
      defmodule BenchmarkSystem do
        def __agents__, do: [%{conserves: [:energy]}]
        def __interactions__, do: [%{body: {:simple, [], []}}]
      end

      result = AII.benchmark(BenchmarkSystem, steps: 1, iterations: 1)

      assert is_map(result)
      assert result.iterations == 1
      assert result.steps == 1
      assert is_number(result.avg_time_ms)
      assert is_number(result.min_time_ms)
      assert is_number(result.max_time_ms)
      assert is_number(result.throughput)
    end
  end

  describe "dispatch_interaction/1" do
    test "dispatches simple interaction" do
      interaction = %{body: {:nearby, [], []}}
      result = AII.dispatch_interaction(interaction)
      assert {:ok, _hardware} = result
    end

    test "dispatches with fallback" do
      interaction = %{body: {:unknown_op, [], []}}
      result = AII.dispatch_interaction(interaction)
      assert {:ok, :cpu} = result  # Should fall back to CPU
    end
  end

  describe "verify_conservation/2" do
    test "verifies conservation for simple case" do
      interaction = %{body: {:simple, [], []}}
      agents = [%{conserves: [:energy]}]

      result = AII.verify_conservation(interaction, agents)
      assert {:needs_runtime_check, _, _} = result
    end
  end

  describe "generate_code/2" do
    test "generates code for different hardware" do
      interaction = %{body: {:simple, [], []}}

      code = AII.generate_code(interaction, :cpu)
      assert is_binary(code)
      assert String.length(code) > 0

      code = AII.generate_code(interaction, :gpu)
      assert is_binary(code)
      assert String.contains?(code, "Generic GPU")
    end
  end

  describe "integration test" do
    test "full pipeline works together" do
      # Test that all components work together
      hardware = AII.available_hardware()
      assert is_list(hardware)

      hint = AII.performance_hint(:cpu)
      assert hint == 1.0

      info = AII.system_info()
      assert info.version == "0.1.0"

      particle = AII.create_particle(mass: 5.0)
      assert particle.mass == 5.0
    end
  end
end
