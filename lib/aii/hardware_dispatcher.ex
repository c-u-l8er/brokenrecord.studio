defmodule AII.HardwareDispatcher do
  @moduledoc """
  Analyzes interaction AST and selects optimal hardware accelerator.
  Maps physics operations to specialized compute units for maximum performance.

  Supports automatic dispatch with fallback chains for robust execution.
  """

  @type hardware :: :auto | :rt_cores | :tensor_cores | :npu | :cuda_cores | :gpu | :cpu | :parallel | :simd
  @type fallback_chain :: [hardware()]
  @type dispatch_result :: {:ok, hardware()} | {:error, term()}

  # Hardware capability detection
  # Hardware capability detection - will be replaced with actual detection
  @hw_types [:cpu, :parallel, :simd, :gpu, :cuda_cores, :rt_cores, :tensor_cores, :npu]

  @doc """
  Dispatches an interaction to the optimal hardware accelerator.

  ## Parameters
  - `interaction`: The interaction AST/map to analyze
  - `fallback`: Fallback strategy (:auto for automatic, or custom chain)

  ## Returns
  - `{:ok, hardware}` - Selected accelerator
  - `{:error, reason}` - Dispatch failed
  """
  @spec dispatch(map(), :auto | fallback_chain()) :: dispatch_result()
  def dispatch(interaction, fallback \\ :auto)

  def dispatch(interaction, :auto) do
    # Analyze interaction to find optimal hardware
    optimal = analyze_interaction(interaction)

    # Check if optimal hardware is available
    if has_hardware?(optimal) do
      {:ok, optimal}
    else
      # Fall back to available hardware in priority order
      fallback_chain = [:rt_cores, :tensor_cores, :npu, :cuda_cores, :gpu, :parallel, :simd, :cpu]
      chain_dispatch(interaction, fallback_chain)
    end
  end

  def dispatch(interaction, fallback_chain) when is_list(fallback_chain) do
    chain_dispatch(interaction, fallback_chain)
  end

  @doc """
  Checks if specific hardware is available on this system.
  """
  @spec has_hardware?(hardware()) :: boolean()
  def has_hardware?(:cpu), do: true
  def has_hardware?(:parallel), do: System.schedulers_online() > 1
  def has_hardware?(:simd), do: has_simd?()
  def has_hardware?(:gpu), do: has_gpu?()
  def has_hardware?(:cuda_cores), do: has_cuda?()
  def has_hardware?(:rt_cores), do: has_rt_cores?()
  def has_hardware?(:tensor_cores), do: has_tensor_cores?()
  def has_hardware?(:npu), do: has_npu?()
  def has_hardware?(_), do: false

  @doc """
  Gets all available hardware on this system.
  """
  @spec available_hardware() :: [hardware()]
  def available_hardware do
    @hw_types
    |> Enum.filter(&has_hardware?/1)
  end

  @doc """
  Analyzes interaction AST to determine optimal hardware.

  ## Analysis Rules:
  - RT Cores: Spatial queries, collision detection, nearest neighbors
  - Tensor Cores: Matrix operations, linear algebra, neural networks
  - NPU: Learned dynamics, inference, pattern recognition
  - CUDA: General parallel computation, physics simulation
  - GPU: Vendor-agnostic GPU compute
  - Parallel: Multi-core CPU operations
  - SIMD: Vectorized operations
  - CPU: Fallback for everything
  """
  @spec analyze_interaction(map()) :: hardware()
  def analyze_interaction(interaction) do
    cond do
      has_spatial_query?(interaction) -> :rt_cores
      has_matrix_operation?(interaction) -> :tensor_cores
      has_learned_model?(interaction) -> :npu
      has_multi_core_cpu?(interaction) -> :parallel
      has_parallel_compute?(interaction) -> :cuda_cores
      has_general_gpu?(interaction) -> :gpu
      has_vector_ops?(interaction) -> :simd
      true -> :cpu
    end
  end

  # Private functions

  defp chain_dispatch(_interaction, []), do: {:error, :no_hardware_available}

  defp chain_dispatch(interaction, [hw | rest]) do
    if has_hardware?(hw) do
      {:ok, hw}
    else
      chain_dispatch(interaction, rest)
    end
  end

  # Hardware detection functions

  defp has_simd? do
    # Check for AVX, AVX2, AVX-512, NEON, etc.
    # For now, assume x86_64 has SIMD
    arch = :erlang.system_info(:system_architecture) |> List.to_string()
    String.starts_with?(arch, "x86_64") or String.starts_with?(arch, "aarch64")
  end

  defp has_gpu? do
    # Check for any GPU
    # This would need platform-specific detection
    # For now, stub as available on most systems
    true
  end

  defp has_cuda? do
    # Check for NVIDIA GPU with CUDA
    # Would need to query GPU capabilities
    false  # Stub - implement proper detection
  end

  defp has_rt_cores? do
    # Check for RT cores (NVIDIA RTX series)
    false  # Stub - implement proper detection
  end

  defp has_tensor_cores? do
    # Check for tensor cores (NVIDIA Volta+)
    false  # Stub - implement proper detection
  end

  defp has_npu? do
    # Check for Neural Processing Unit
    # Apple Neural Engine, Qualcomm Hexagon, etc.
    case :os.type() do
      {:unix, :darwin} -> true  # Assume Apple Silicon has ANE
      _ -> false
    end
  end

  # AST analysis functions

  @spatial_keywords [:nearby, :colliding?, :within_radius, :find_neighbors,
                     :spatial_query, :ray_cast, :collision, :bvh, :octree]

  @matrix_keywords [:matrix_multiply, :dot_product, :matmul, :outer_product,
                    :linear_transform, :tensor_op, :gemm, :blas]

  @neural_keywords [:predict, :infer, :neural_network, :forward_pass,
                   :model_eval, :embedding, :attention, :transformer]

  @parallel_keywords [:reduce, :scan, :flow, :task_async]

  @gpu_keywords [:gpu_compute, :shader, :compute_shader, :kernel]

  @cpu_parallel_keywords [:flow_map, :task_async, :parallel_stream, :concurrent, :parallel_map]

  @simd_keywords [:vector_add, :vector_mul, :simd_map, :vectorized, :avx]

  defp has_spatial_query?(interaction) do
    ast_contains?(interaction, @spatial_keywords)
  end

  defp has_matrix_operation?(interaction) do
    ast_contains?(interaction, @matrix_keywords)
  end

  defp has_learned_model?(interaction) do
    ast_contains?(interaction, @neural_keywords)
  end

  defp has_parallel_compute?(interaction) do
    ast_contains?(interaction, @parallel_keywords)
  end

  defp has_general_gpu?(interaction) do
    ast_contains?(interaction, @gpu_keywords)
  end

  defp has_multi_core_cpu?(interaction) do
    ast_contains?(interaction, @cpu_parallel_keywords)
  end

  defp has_vector_ops?(interaction) do
    ast_contains?(interaction, @simd_keywords)
  end

  # AST traversal to find keywords
  defp ast_contains?(ast, keywords) when is_map(ast) do
    # Check interaction body or AST
    body = Map.get(ast, :body) || Map.get(ast, :ast) || ast

    walk_ast(body, keywords)
  end

  defp ast_contains?(ast, keywords) do
    walk_ast(ast, keywords)
  end

  defp walk_ast(ast, keywords) do
    Macro.prewalk(ast, false, fn
      {:., _, [{call, _, _}, _]} = node, acc ->
        if call in keywords do
          {node, true}
        else
          {node, acc}
        end
      {call, _, _} = node, acc ->
        if call in keywords do
          {node, true}
        else
          {node, acc}
        end
      node, acc -> {node, acc}
    end)
    |> elem(1)
  end

  @doc """
  Performance hints for different hardware types.

  Returns estimated speedup factor relative to CPU.
  """
  @spec performance_hint(hardware()) :: float()
  def performance_hint(:cpu), do: 1.0
  def performance_hint(:simd), do: 4.0
  def performance_hint(:parallel), do: System.schedulers_online() * 0.8
  def performance_hint(:gpu), do: 50.0
  def performance_hint(:cuda_cores), do: 100.0
  def performance_hint(:tensor_cores), do: 200.0
  def performance_hint(:rt_cores), do: 150.0
  def performance_hint(:npu), do: 300.0
  def performance_hint(_), do: 1.0

  @doc """
  Memory requirements hint for different hardware.

  Returns estimated memory overhead factor.
  """
  @spec memory_hint(hardware()) :: float()
  def memory_hint(:cpu), do: 1.0
  def memory_hint(:simd), do: 1.0
  def memory_hint(:parallel), do: 1.2
  def memory_hint(:gpu), do: 2.0
  def memory_hint(:cuda_cores), do: 2.0
  def memory_hint(:tensor_cores), do: 3.0
  def memory_hint(:rt_cores), do: 4.0
  def memory_hint(:npu), do: 1.5
  def memory_hint(_), do: 1.0

  @doc """
  Power efficiency hint (higher = more efficient).

  Returns power efficiency score (operations per watt).
  """
  @spec efficiency_hint(hardware()) :: float()
  def efficiency_hint(:cpu), do: 1.0
  def efficiency_hint(:simd), do: 1.5
  def efficiency_hint(:parallel), do: 1.2
  def efficiency_hint(:gpu), do: 0.5
  def efficiency_hint(:cuda_cores), do: 0.6
  def efficiency_hint(:tensor_cores), do: 0.8
  def efficiency_hint(:rt_cores), do: 0.7
  def efficiency_hint(:npu), do: 2.0
  def efficiency_hint(_), do: 1.0
end
