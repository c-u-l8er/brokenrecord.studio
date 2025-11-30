defmodule AII.CodegenTest do
  use ExUnit.Case
  alias AII.Codegen

  describe "generate/2" do
    test "generates RT cores code" do
      interaction = %{body: {:nearby, [], []}}
      code = Codegen.generate(interaction, :rt_cores)

      assert String.contains?(code, "Vulkan Ray Tracing")
      assert String.contains?(code, "rayQueryEXT")
      assert String.contains?(code, "buildBVH")
    end

    test "generates tensor cores code" do
      interaction = %{body: {:matrix_multiply, [], []}}
      code = Codegen.generate(interaction, :tensor_cores)

      assert String.contains?(code, "cooperative_matrix")
      assert String.contains?(code, "coopMatMulAdd")
      assert String.contains?(code, "Tensor Cores")
    end

    test "generates NPU code" do
      interaction = %{body: {:predict, [], []}}
      code = Codegen.generate(interaction, :npu)

      assert String.contains?(code, "NPU Inference")
      assert String.contains?(code, "__APPLE__")
      assert String.contains?(code, "Core ML")
      assert String.contains?(code, "__ANDROID__")
      assert String.contains?(code, "NNAPI")
    end

    test "generates CUDA cores code" do
      interaction = %{body: {:parallel_map, [], []}}
      code = Codegen.generate(interaction, :cuda_cores)

      assert String.contains?(code, "__global__")
      assert String.contains?(code, "cudaMalloc")
      assert String.contains?(code, "aii_kernel")
    end

    test "generates generic GPU code" do
      interaction = %{body: {:gpu_compute, [], []}}
      code = Codegen.generate(interaction, :gpu)

      assert String.contains?(code, "#version 450")
      assert String.contains?(code, "local_size_x = 256")
      assert String.contains?(code, "Compute Shader")
    end

    test "generates parallel CPU code" do
      interaction = %{body: {:flow_map, [], []}}
      code = Codegen.generate(interaction, :parallel)

      assert String.contains?(code, "Flow.from_enumerable")
      assert String.contains?(code, "Flow.partition")
      assert String.contains?(code, "GeneratedParallel")
    end

    test "generates SIMD code" do
      interaction = %{body: {:vector_add, [], []}}
      code = Codegen.generate(interaction, :simd)

      assert String.contains?(code, "__m256")
      assert String.contains?(code, "_mm256_load_ps")
      assert String.contains?(code, "SIMD Vectorized")
    end

    test "generates CPU fallback code" do
      interaction = %{body: {:unknown_op, [], []}}
      code = Codegen.generate(interaction, :cpu)

      assert String.contains?(code, "CPU Fallback")
      assert String.contains?(code, "process_particles_cpu")
      assert String.contains?(code, "GeneratedCPU")
    end
  end

  describe "generate_auto/1" do
    test "auto-generates for spatial queries" do
      interaction = %{name: :nearby, body: {:nearby, [], []}}
      code = Codegen.generate_auto(interaction)

      # Dispatches to RT cores (available on this system)
      assert String.contains?(code, "Vulkan Ray Tracing")
      assert String.contains?(code, "RT Cores")
    end

    test "auto-generates for matrix operations" do
      interaction = %{name: :matrix_multiply, body: {:matrix_multiply, [], []}}
      code = Codegen.generate_auto(interaction)

      # Dispatches to tensor cores (available on this system)
      assert String.contains?(code, "Vulkan Tensor Cores")
      assert String.contains?(code, "cooperative_matrix")
    end

    test "auto-generates for neural operations" do
      interaction = %{name: :predict, body: {:predict, [], []}}
      code = Codegen.generate_auto(interaction)

      # NPU not available, falls back to RT cores (first in chain after NPU)
      assert String.contains?(code, "Vulkan Ray Tracing")
      assert String.contains?(code, "RT Cores")
    end

    test "falls back to CPU for unknown operations" do
      interaction = %{name: :unknown_op, body: {:unknown_op, [], []}}
      code = Codegen.generate_auto(interaction)

      assert String.contains?(code, "CPU Fallback")
    end
  end

  describe "code structure" do
    test "generated code includes interaction info" do
      interaction = %{body: {:test_op, [], []}, name: :test_interaction}
      code = Codegen.generate(interaction, :cpu)

      assert String.contains?(code, "Generated for")
      assert String.contains?(code, inspect(interaction))
    end

    test "different hardware generates different code" do
      interaction = %{body: {:vector_add, [], []}}

      cpu_code = Codegen.generate(interaction, :cpu)
      simd_code = Codegen.generate(interaction, :simd)
      gpu_code = Codegen.generate(interaction, :gpu)

      refute cpu_code == simd_code
      refute cpu_code == gpu_code
      refute simd_code == gpu_code

      assert String.contains?(cpu_code, "CPU Fallback")
      assert String.contains?(simd_code, "__m256")
      assert String.contains?(gpu_code, "#version 450")
    end
  end

  describe "error handling" do
    test "handles empty interaction" do
      interaction = %{}
      code = Codegen.generate(interaction, :cpu)

      assert is_binary(code)
      assert String.length(code) > 0
    end

    test "handles nil interaction" do
      # This should not crash
      code = Codegen.generate(nil, :cpu)

      assert is_binary(code)
    end
  end

  describe "performance considerations" do
    test "generated code includes comments about performance" do
      interaction = %{body: {:matrix_multiply, [], []}}
      code = Codegen.generate(interaction, :tensor_cores)

      assert String.contains?(code, "Tensor cores execute")
    end

    test "parallel code includes Flow usage" do
      interaction = %{body: {:parallel_map, [], []}}
      code = Codegen.generate(interaction, :parallel)

      assert String.contains?(code, "Flow.from_enumerable")
      assert String.contains?(code, "Flow.map")
    end
  end
end
