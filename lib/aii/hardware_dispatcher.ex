defmodule AII.HardwareDispatcher do
  @moduledoc """
  Analyzes interaction AST and selects optimal hardware accelerator.
  Maps physics operations to specialized compute units for maximum performance.

  Supports automatic dispatch with fallback chains for robust execution.
  """

  alias AII.HardwareDetection

  @type hardware :: :auto | :rt_cores | :tensor_cores | :npu | :cuda_cores | :gpu | :cpu | :parallel | :simd
  @type fallback_chain :: [hardware()]
  @type dispatch_result :: {:ok, hardware()} | {:error, term()}

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
    # Check if interaction has explicit accelerator hint
    case Map.get(interaction, :accelerator, :auto) do
      :auto ->
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
      explicit_hw ->
        # Use explicit accelerator hint
        if has_hardware?(explicit_hw) do
          {:ok, explicit_hw}
        else
          {:error, "Requested hardware #{explicit_hw} not available"}
        end
    end
  end

  def dispatch(interaction, fallback_chain) when is_list(fallback_chain) do
    chain_dispatch(interaction, fallback_chain)
  end

  @doc """
  Checks if specific hardware is available on this system.
  """
  @spec has_hardware?(hardware()) :: boolean()
  def has_hardware?(hw) do
    case hw do
      :cpu -> true
      :parallel -> HardwareDetection.detect().core_count > 1
      :simd -> HardwareDetection.detect().simd_avx2 or HardwareDetection.detect().simd_avx512 or HardwareDetection.detect().simd_neon
      :gpu -> HardwareDetection.detect().gpu_count > 0
      :cuda_cores -> HardwareDetection.detect().cuda
      :rt_cores -> HardwareDetection.detect().rt_cores
      :tensor_cores -> HardwareDetection.detect().tensor_cores
      :npu -> HardwareDetection.detect().npu
      _ -> false
    end
  end

  @doc """
  Gets all available hardware on this system.
  """
  @spec available_hardware() :: [hardware()]
  def available_hardware do
    caps = HardwareDetection.detect()

    [:cpu] ++
    (if caps.core_count > 1, do: [:parallel], else: []) ++
    (if caps.simd_avx2 or caps.simd_avx512 or caps.simd_neon, do: [:simd], else: []) ++
    (if caps.gpu_count > 0, do: [:gpu], else: []) ++
    (if caps.cuda, do: [:cuda_cores], else: []) ++
    (if caps.rt_cores, do: [:rt_cores], else: []) ++
    (if caps.tensor_cores, do: [:tensor_cores], else: []) ++
    (if caps.npu, do: [:npu], else: [])
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
      # Hardware priority order: RT cores > Tensor cores > NPU > GPU > CUDA > SIMD > CPU
      has_collision_detection?(interaction) -> :rt_cores
      has_spatial_query?(interaction) -> :rt_cores
      has_matrix_operation?(interaction) -> :tensor_cores
      has_learned_model?(interaction) -> :npu
      has_general_gpu?(interaction) -> :gpu
      has_parallel_compute?(interaction) -> :cuda_cores
      has_multi_core_cpu?(interaction) -> :parallel
      # SIMD for vectorizable physics operations
      is_vectorizable_physics?(interaction) -> :simd
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



  # AST analysis functions

  @spatial_keywords [:nearby, :colliding?, :within_radius, :find_neighbors,
                     :spatial_query, :ray_cast, :collision, :bvh, :octree]

  @collision_keywords [:collision, :colliding?, :collide, :intersect, :overlap, :hit_test]

  @matrix_keywords [:matrix_multiply, :dot_product, :dot, :matmul, :outer_product,
                    :linear_transform, :tensor_op, :gemm, :blas]

  @neural_keywords [:predict, :infer, :neural_network, :forward_pass,
                   :model_eval, :embedding, :attention, :transformer]

  @parallel_keywords [:reduce, :scan, :flow]

  @gpu_keywords [:gpu_compute, :shader, :compute_shader, :kernel]

  @cpu_parallel_keywords [:flow_map, :task_async, :parallel_stream, :concurrent, :parallel_map]

  @simd_keywords [:vector_add, :vector_mul, :simd_map, :vectorized, :avx]

  # Physics operations that can be vectorized with SIMD
  @vectorizable_keywords [:position, :velocity, :integrate, :update, :force, :acceleration, :momentum]

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

  # Check if interaction contains vectorizable physics operations
  defp is_vectorizable_physics?(interaction) do
    ast_contains?(interaction, @vectorizable_keywords)
  end

  # Check if interaction contains collision detection operations
  defp has_collision_detection?(interaction) do
    # Check interaction name first (more reliable than AST search)
    interaction.name in [:detect_collisions, :gravitational_collisions, :conserved_collisions] or
    ast_contains?(interaction, @collision_keywords)
  end

  # Check if interaction is suitable for GPU compute
  defp has_gpu_compute?(interaction) do
    # GPU is good for parallel workloads and vector operations
    has_parallel_compute?(interaction) or has_vector_ops?(interaction) or is_vectorizable_physics?(interaction)
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
