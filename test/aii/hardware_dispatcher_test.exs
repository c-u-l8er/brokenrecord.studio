

defmodule AII.HardwareDispatcherTest do
  use ExUnit.Case
  alias AII.HardwareDispatcher

  describe "dispatch/2" do
    test "auto dispatch with default chain" do
      interaction = %{body: {:nearby, [], []}}
      {:ok, hardware} = HardwareDispatcher.dispatch(interaction, :auto)
      assert hardware in [:rt_cores, :tensor_cores, :npu, :cuda_cores, :gpu, :parallel, :simd, :cpu]
    end

    test "custom fallback chain" do
      interaction = %{body: {:matrix_multiply, [], []}}
      chain = [:tensor_cores, :cuda_cores, :cpu]
      {:ok, hardware} = HardwareDispatcher.dispatch(interaction, chain)
      assert hardware in chain
    end

    test "fallback when hardware unavailable" do
      interaction = %{body: {:some_op, [], []}}
      chain = [:nonexistent_hw, :cpu]
      {:ok, hardware} = HardwareDispatcher.dispatch(interaction, chain)
      assert hardware == :cpu
    end

    test "error when no hardware available" do
      interaction = %{body: {:some_op, [], []}}
      chain = [:nonexistent_hw1, :nonexistent_hw2]
      assert {:error, :no_hardware_available} = HardwareDispatcher.dispatch(interaction, chain)
    end
  end

  describe "has_hardware?/1" do
    test "cpu always available" do
      assert HardwareDispatcher.has_hardware?(:cpu) == true
    end

    test "parallel available on multi-core systems" do
      schedulers = System.schedulers_online()
      expected = schedulers > 1
      assert HardwareDispatcher.has_hardware?(:parallel) == expected
    end

    test "simd available on supported architectures" do
      arch = :erlang.system_info(:system_architecture) |> List.to_string()
      expected = String.starts_with?(arch, "x86_64") or String.starts_with?(arch, "aarch64")
      assert HardwareDispatcher.has_hardware?(:simd) == expected
    end

    test "gpu assumed available" do
      assert HardwareDispatcher.has_hardware?(:gpu) == true
    end

    test "specialized hardware not available (stubs)" do
      refute HardwareDispatcher.has_hardware?(:rt_cores)
      refute HardwareDispatcher.has_hardware?(:tensor_cores)
      refute HardwareDispatcher.has_hardware?(:cuda_cores)
    end

    test "npu available on macOS" do
      expected = case :os.type() do
        {:unix, :darwin} -> true
        _ -> false
      end
      assert HardwareDispatcher.has_hardware?(:npu) == expected
    end
  end

  describe "available_hardware/0" do
    test "returns list of available hardware" do
      available = HardwareDispatcher.available_hardware()
      assert is_list(available)
      assert :cpu in available
      assert :parallel in available or System.schedulers_online() <= 1
      assert :simd in available
      assert :gpu in available
    end
  end

  describe "analyze_interaction/1" do
    test "spatial queries dispatch to rt_cores" do
      interactions = [
        %{body: {:nearby, [], []}},
        %{body: {:colliding?, [], []}},
        %{body: {:within_radius, [], []}},
        %{body: {:find_neighbors, [], []}},
        %{body: {:spatial_query, [], []}},
        %{body: {:ray_cast, [], []}}
      ]

      for interaction <- interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :rt_cores
      end
    end

    test "matrix operations dispatch to tensor_cores" do
      interactions = [
        %{body: {:matrix_multiply, [], []}},
        %{body: {:dot_product, [], []}},
        %{body: {:matmul, [], []}},
        %{body: {:outer_product, [], []}},
        %{body: {:linear_transform, [], []}},
        %{body: {:tensor_op, [], []}}
      ]

      for interaction <- interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :tensor_cores
      end
    end

    test "neural operations dispatch to npu" do
      interactions = [
        %{body: {:predict, [], []}},
        %{body: {:infer, [], []}},
        %{body: {:neural_network, [], []}},
        %{body: {:forward_pass, [], []}},
        %{body: {:model_eval, [], []}}
      ]

      for interaction <- interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :npu
      end
    end

    test "parallel operations dispatch correctly" do
      # parallel_map goes to multi-core CPU
      assert HardwareDispatcher.analyze_interaction(%{body: {:parallel_map, [], []}}) == :parallel

      # reduce and scan go to CUDA cores
      cuda_interactions = [
        %{body: {:reduce, [], []}},
        %{body: {:scan, [], []}}
      ]

      for interaction <- cuda_interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :cuda_cores
      end
    end

    test "gpu operations dispatch to gpu" do
      interactions = [
        %{body: {:gpu_compute, [], []}},
        %{body: {:shader, [], []}},
        %{body: {:compute_shader, [], []}}
      ]

      for interaction <- interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :gpu
      end
    end

    test "multi-core cpu operations dispatch to parallel" do
      interactions = [
        %{body: {:flow_map, [], []}},
        %{body: {:task_async, [], []}},
        %{body: {:parallel_stream, [], []}}
      ]

      for interaction <- interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :parallel
      end
    end

    test "simd operations dispatch to simd" do
      interactions = [
        %{body: {:vector_add, [], []}},
        %{body: {:vector_mul, [], []}},
        %{body: {:simd_map, [], []}},
        %{body: {:vectorized, [], []}}
      ]

      for interaction <- interactions do
        assert HardwareDispatcher.analyze_interaction(interaction) == :simd
      end
    end

    test "unknown operations dispatch to cpu" do
      interaction = %{body: {:unknown_op, [], []}}
      assert HardwareDispatcher.analyze_interaction(interaction) == :cpu
    end

    test "handles complex AST structures" do
      # Test nested calls
      interaction = %{
        body: {:block, [], [
          {:let, [], [
            {:particle, [], nil},
            {:do, [], [
              {:nearby, [], []},
              {:matrix_multiply, [], []}
            ]}
          ]}
        ]}
      }

      # Should detect spatial query first
      assert HardwareDispatcher.analyze_interaction(interaction) == :rt_cores
    end
  end

  describe "performance hints" do
    test "performance_hint/1 returns expected values" do
      assert HardwareDispatcher.performance_hint(:cpu) == 1.0
      assert HardwareDispatcher.performance_hint(:simd) == 4.0
      assert HardwareDispatcher.performance_hint(:gpu) == 50.0
      assert HardwareDispatcher.performance_hint(:cuda_cores) == 100.0
      assert HardwareDispatcher.performance_hint(:tensor_cores) == 200.0
      assert HardwareDispatcher.performance_hint(:rt_cores) == 150.0
      assert HardwareDispatcher.performance_hint(:npu) == 300.0
    end

    test "parallel hint scales with CPU cores" do
      expected = System.schedulers_online() * 0.8
      assert HardwareDispatcher.performance_hint(:parallel) == expected
    end
  end

  describe "memory hints" do
    test "memory_hint/1 returns expected values" do
      assert HardwareDispatcher.memory_hint(:cpu) == 1.0
      assert HardwareDispatcher.memory_hint(:simd) == 1.0
      assert HardwareDispatcher.memory_hint(:parallel) == 1.2
      assert HardwareDispatcher.memory_hint(:gpu) == 2.0
      assert HardwareDispatcher.memory_hint(:cuda_cores) == 2.0
      assert HardwareDispatcher.memory_hint(:tensor_cores) == 3.0
      assert HardwareDispatcher.memory_hint(:rt_cores) == 4.0
      assert HardwareDispatcher.memory_hint(:npu) == 1.5
    end
  end

  describe "efficiency hints" do
    test "efficiency_hint/1 returns expected values" do
      assert HardwareDispatcher.efficiency_hint(:cpu) == 1.0
      assert HardwareDispatcher.efficiency_hint(:simd) == 1.5
      assert HardwareDispatcher.efficiency_hint(:parallel) == 1.2
      assert HardwareDispatcher.efficiency_hint(:gpu) == 0.5
      assert HardwareDispatcher.efficiency_hint(:cuda_cores) == 0.6
      assert HardwareDispatcher.efficiency_hint(:tensor_cores) == 0.8
      assert HardwareDispatcher.efficiency_hint(:rt_cores) == 0.7
      assert HardwareDispatcher.efficiency_hint(:npu) == 2.0
    end
  end

  describe "AST analysis" do
    test "ast_contains? detects keywords in simple calls" do
      # This is testing private function, but we can test through public API
      interaction = %{body: {:nearby, [], []}}
      assert HardwareDispatcher.analyze_interaction(interaction) == :rt_cores
    end

    test "ast_contains? handles function calls with modules" do
      interaction = %{body: {{:., [], [{:Enum, [], nil}, :map]}, [], []}}
      # This should not match any hardware-specific keywords
      assert HardwareDispatcher.analyze_interaction(interaction) == :cpu
    end

    test "ast_contains? works with nested structures" do
      interaction = %{
        body: {:if, [], [
          {:nearby, [], []},
          {:do_something, [], []},
          {:else, [], []}
        ]}
      }
      assert HardwareDispatcher.analyze_interaction(interaction) == :rt_cores
    end
  end
end
