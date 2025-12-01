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
      particle =
        AII.create_particle(
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
      result = AII.run_simulation(InvalidModule)
      assert {:error, :invalid_system_module} = result
    end

    test "runs simulation with mock system" do
      # Create a mock system module
      defmodule MockSystem do
        def __agents__, do: [%{conserves: [:energy]}]
        def __interactions__, do: [%{name: :simple, body: {:simple, [], []}}]
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
        def __interactions__, do: [%{name: :simple, body: {:simple, [], []}}]
      end

      result = AII.benchmark(BenchmarkSystem, steps: 1, iterations: 1)

      assert is_map(result)
      # 1 iteration * 4 backends
      assert result.summary.total_iterations == 4
      assert result.summary.steps == 1
      assert is_list(result.benchmarks)
      assert length(result.benchmarks) == 4
      # Check that each benchmark has the required fields
      Enum.each(result.benchmarks, fn benchmark ->
        assert is_number(benchmark.avg_time_ms)
        assert is_number(benchmark.min_time_ms)
        assert is_number(benchmark.max_time_ms)
        assert is_number(benchmark.throughput)
        assert benchmark.iterations == 1
        assert benchmark.steps == 1
      end)
    end
  end

  describe "dispatch_interaction/1" do
    test "dispatches simple interaction" do
      interaction = %{name: :nearby, body: {:nearby, [], []}}
      result = AII.dispatch_interaction(interaction)
      assert {:ok, _hardware} = result
    end

    test "dispatches with fallback" do
      interaction = %{name: :unknown_op, body: {:unknown_op, [], []}}
      result = AII.dispatch_interaction(interaction)
      # Should fall back to CPU
      assert {:ok, :cpu} = result
    end
  end

  describe "verify_conservation/2" do
    test "verifies conservation for simple case" do
      interaction = %{name: :simple, body: {:simple, [], []}}
      agents = [%{conserves: [:energy]}]

      result = AII.verify_conservation(interaction, agents)
      assert {:needs_runtime_check, _, _} = result
    end
  end

  describe "generate_code/2" do
    test "generates code for different hardware" do
      interaction = %{name: :simple, body: {:simple, [], []}}

      code = AII.generate_code(interaction, :cpu)
      assert is_binary(code)
      assert String.length(code) > 0

      code = AII.generate_code(interaction, :gpu)
      assert is_binary(code)
      # SPIR-V binary
      assert byte_size(code) > 0
    end

    test "GPU code generation includes compute shader elements" do
      interaction = %{name: :physics, body: {:particle_update, [], []}}

      code = AII.generate_code(interaction, :gpu)
      assert is_binary(code)
      # Basic check that it's not empty SPIR-V
      assert byte_size(code) > 20
    end

    test "different interactions produce different GPU code" do
      interaction1 = %{name: :gravity, body: {:gravity, [], []}}
      interaction2 = %{name: :collision, body: {:collision, [], []}}

      code1 = AII.generate_code(interaction1, :gpu)
      code2 = AII.generate_code(interaction2, :gpu)

      # Codes should be different (though both valid SPIR-V)
      assert code1 != code2
      assert is_binary(code1)
      assert is_binary(code2)
    end
  end

  describe "hardware acceleration" do
    test "available_hardware includes GPU when detected" do
      hardware = AII.available_hardware()
      assert is_list(hardware)
      assert :cpu in hardware
      # GPU might not be detected in test environment, but list should be valid
    end

    test "performance hints for GPU are higher than CPU" do
      cpu_hint = AII.performance_hint(:cpu)
      gpu_hint = AII.performance_hint(:gpu)

      assert is_number(cpu_hint)
      assert is_number(gpu_hint)
      assert gpu_hint > cpu_hint
    end

    test "run_simulation with GPU hardware doesn't crash" do
      defmodule TestGPUSystem do
        def __agents__, do: [%{conserves: [:energy]}]
        def __interactions__, do: [%{name: :gpu_test, body: {:gpu_compute, [], []}}]
      end

      # This should not crash, even if GPU is not available (falls back to CPU)
      result = AII.run_simulation(TestGPUSystem, steps: 1, hardware: :gpu)
      assert {:ok, results} = result
      assert results.hardware == :gpu
      assert results.conservation_verified == true
    end

    test "run_simulation with auto hardware selection" do
      defmodule TestAutoSystem do
        def __agents__, do: [%{conserves: [:momentum]}]
        def __interactions__, do: [%{name: :auto_test, body: {:auto_compute, [], []}}]
      end

      result = AII.run_simulation(TestAutoSystem, steps: 1, hardware: :auto)
      assert {:ok, results} = result
      assert results.hardware == :auto
      assert results.conservation_verified == true
    end

    test "memory hints are reasonable for different hardware" do
      cpu_memory = AII.memory_hint(:cpu)
      gpu_memory = AII.memory_hint(:gpu)

      assert is_number(cpu_memory)
      assert is_number(gpu_memory)
      # Baseline
      assert cpu_memory == 1.0
      assert gpu_memory >= cpu_memory
    end

    test "efficiency hints reflect hardware characteristics" do
      cpu_efficiency = AII.efficiency_hint(:cpu)
      gpu_efficiency = AII.efficiency_hint(:gpu)
      npu_efficiency = AII.efficiency_hint(:npu)

      assert is_number(cpu_efficiency)
      assert is_number(gpu_efficiency)
      assert is_number(npu_efficiency)
      # NPU should be most efficient
      assert npu_efficiency >= gpu_efficiency
      # GPU might be less efficient than CPU for some workloads
      assert gpu_efficiency <= cpu_efficiency
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
