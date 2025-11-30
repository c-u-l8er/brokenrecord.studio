defmodule AII.HardwareDetection do
  @moduledoc """
  Comprehensive hardware capability detection for AII.

  Detects available accelerators including Vulkan, CUDA, Metal, OpenCL,
  SIMD instructions, and Neural Processing Units (NPUs).
  """

  require Logger

  @type hardware_caps :: %{
          vulkan: boolean(),
          cuda: boolean(),
          metal: boolean(),
          opencl: boolean(),
          rt_cores: boolean(),
          tensor_cores: boolean(),
          npu: boolean(),
          simd_avx2: boolean(),
          simd_avx512: boolean(),
          simd_neon: boolean(),
          core_count: non_neg_integer(),
          gpu_count: non_neg_integer(),
          npu_count: non_neg_integer()
        }

  @doc """
  Detects all available hardware capabilities on the current system.

  Returns a map with detailed hardware information.
  """
  @spec detect() :: hardware_caps()
  def detect do
    %{
      vulkan: detect_vulkan(),
      cuda: detect_cuda(),
      metal: detect_metal(),
      opencl: detect_opencl(),
      rt_cores: detect_rt_cores(),
      tensor_cores: detect_tensor_cores(),
      npu: detect_npu(),
      simd_avx2: detect_simd_avx2(),
      simd_avx512: detect_simd_avx512(),
      simd_neon: detect_simd_neon(),
      core_count: System.schedulers_online(),
      gpu_count: detect_gpu_count(),
      npu_count: detect_npu_count()
    }
  end

  @doc """
  Checks if RT cores are available (NVIDIA RTX series).
  """
  @spec has_rt_cores?() :: boolean()
  def has_rt_cores? do
    detect_rt_cores()
  end

  @doc """
  Checks if tensor cores are available (NVIDIA Volta+).
  """
  @spec has_tensor_cores?() :: boolean()
  def has_tensor_cores? do
    detect_tensor_cores()
  end

  @doc """
  Checks if NPU is available.
  """
  @spec has_npu?() :: boolean()
  def has_npu? do
    detect_npu()
  end

  # Private detection functions

  @spec detect_vulkan() :: boolean()
  defp detect_vulkan do
    # Try to load Vulkan library
    try do
      # Check if Vulkan loader is available
      case System.find_executable("vulkaninfo") do
        nil -> false

        _path ->
          # Run vulkaninfo with timeout to check if Vulkan works
          task =
            Task.async(fn ->
              System.cmd("vulkaninfo", ["--summary"], stderr_to_stdout: true)
            end)

          case Task.yield(task, 2000) do
            {:ok, {_output, 0}} ->
              true

            _ ->
              Task.shutdown(task, :brutal_kill)
              false
          end
      end
    catch
      _ -> false
    end
  end

  @spec detect_cuda() :: boolean()
  defp detect_cuda do
    # Check for NVIDIA CUDA toolkit
    case System.find_executable("nvidia-smi") do
      nil ->
        false

      _path ->
        # Check if CUDA runtime is available with timeout
        task =
          Task.async(fn ->
            System.cmd("nvidia-smi", [], stderr_to_stdout: true)
          end)

        case Task.yield(task, 2000) do
          {:ok, {_output, 0}} ->
            true

          _ ->
            Task.shutdown(task, :brutal_kill)
            false
        end
    end
  end

  @spec detect_metal() :: boolean()
  defp detect_metal do
    # Metal is only available on macOS
    case :os.type() do
      {:unix, :darwin} ->
        # Check for Metal framework (simplified check)
        File.exists?("/System/Library/Frameworks/Metal.framework")

      _ ->
        false
    end
  end

  @spec detect_opencl() :: boolean()
  defp detect_opencl do
    # Check for OpenCL ICD loader
    case System.find_executable("clinfo") do
      nil ->
        false

      _path ->
        # Check OpenCL with timeout
        task =
          Task.async(fn ->
            System.cmd("clinfo", [], stderr_to_stdout: true)
          end)

        case Task.yield(task, 2000) do
          {:ok, {_output, 0}} ->
            true

          _ ->
            Task.shutdown(task, :brutal_kill)
            false
        end
    end
  end

  @spec detect_rt_cores() :: boolean()
  defp detect_rt_cores do
    # RT cores require Vulkan or CUDA with RTX hardware
    (detect_vulkan() and has_rtx_hardware?()) or (detect_cuda() and has_rtx_hardware?())
  end

  @spec detect_tensor_cores() :: boolean()
  defp detect_tensor_cores do
    # Tensor cores available on NVIDIA Volta+ GPUs
    detect_cuda() and has_volta_plus_hardware?()
  end

  @spec detect_npu() :: boolean()
  defp detect_npu do
    case :os.type() do
      {:unix, :darwin} ->
        # Apple Neural Engine (ANE) on Apple Silicon
        detect_apple_neural_engine()

      {:unix, :linux} ->
        # Check for various NPUs
        detect_linux_npu()

      _ ->
        false
    end
  end

  @spec detect_simd_avx2() :: boolean()
  defp detect_simd_avx2 do
    # Check CPU features for AVX2
    # First try Erlang's cpu_topology
    case :erlang.system_info(:cpu_topology) do
      cpus when is_list(cpus) ->
        # Check if any CPU has AVX2
        has_avx2 =
          Enum.any?(cpus, fn cpu ->
            case cpu do
              %{available: features} when is_list(features) ->
                :avx2 in features

              _ ->
                false
            end
          end)

        if has_avx2, do: true, else: fallback_avx2_detection()

      _ ->
        fallback_avx2_detection()
    end
  end

  @spec fallback_avx2_detection() :: boolean()
  defp fallback_avx2_detection do
    # Fallback: assume AVX2 is available on modern x86_64 systems
    arch = :erlang.system_info(:system_architecture) |> List.to_string()
    String.contains?(arch, "x86_64") or String.contains?(arch, "amd64")
  end

  @spec detect_simd_avx512() :: boolean()
  defp detect_simd_avx512 do
    # Check for AVX-512 support
    case :erlang.system_info(:cpu_topology) do
      cpus when is_list(cpus) ->
        has_avx512 =
          Enum.any?(cpus, fn cpu ->
            case cpu do
              %{available: features} when is_list(features) ->
                :avx512 in features

              _ ->
                false
            end
          end)

        # No fallback for AVX-512 as it's less common
        if has_avx512, do: true, else: false

      _ ->
        false
    end
  end

  @spec detect_simd_neon() :: boolean()
  defp detect_simd_neon do
    # NEON is available on ARM64
    arch = :erlang.system_info(:system_architecture) |> List.to_string()
    String.contains?(arch, "aarch64") or String.contains?(arch, "arm64")
  end

  @spec detect_gpu_count() :: non_neg_integer()
  defp detect_gpu_count do
    # Count available GPUs
    cond do
      detect_vulkan() ->
        # Use Vulkan to count GPUs
        count_vulkan_gpus()

      detect_cuda() ->
        # Use NVIDIA tools
        count_cuda_gpus()

      detect_opencl() ->
        # Use OpenCL
        count_opencl_devices()

      true ->
        0
    end
  end

  @spec detect_npu_count() :: non_neg_integer()
  defp detect_npu_count do
    if detect_npu() do
      case :os.type() do
        # Typically one ANE per SoC
        {:unix, :darwin} -> 1
        {:unix, :linux} -> count_linux_npus()
        _ -> 0
      end
    else
      0
    end
  end

  # Helper functions

  @spec has_rtx_hardware?() :: boolean()
  defp has_rtx_hardware? do
    # Check for RTX GPUs via nvidia-smi
    task =
      Task.async(fn ->
        System.cmd("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"],
          stderr_to_stdout: true
        )
      end)

    case Task.yield(task, 2000) do
      {:ok, {output, 0}} ->
        String.contains?(output, "RTX") or String.contains?(output, "GeForce RTX")

      _ ->
        Task.shutdown(task, :brutal_kill)
        # Fallback: assume RTX if we have CUDA (most modern NVIDIA GPUs have RTX)
        detect_cuda()
    end
  end

  @spec has_volta_plus_hardware?() :: boolean()
  defp has_volta_plus_hardware? do
    # Check for Volta+ GPUs
    task =
      Task.async(fn ->
        System.cmd("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"],
          stderr_to_stdout: true
        )
      end)

    case Task.yield(task, 2000) do
      {:ok, {output, 0}} ->
        # List of Volta+ GPU names
        volta_plus = ["Tesla V100", "Tesla T4", "GeForce RTX", "A100", "H100", "L40"]
        Enum.any?(volta_plus, fn gpu -> String.contains?(output, gpu) end)

      _ ->
        Task.shutdown(task, :brutal_kill)
        # Fallback: assume Volta+ if we have CUDA (most modern NVIDIA GPUs are Volta+)
        detect_cuda()
    end
  end

  @spec detect_apple_neural_engine() :: boolean()
  defp detect_apple_neural_engine do
    # Check for Apple Neural Engine
    # This is a simplified check - in practice, would need Core ML or similar
    task =
      Task.async(fn ->
        System.cmd("system_profiler", ["SPHardwareDataType"], stderr_to_stdout: true)
      end)

    case Task.yield(task, 2000) do
      {:ok, {output, 0}} ->
        String.contains?(output, "Apple M") or String.contains?(output, "Neural Engine")

      _ ->
        Task.shutdown(task, :brutal_kill)
        false
    end
  end

  @spec detect_linux_npu() :: boolean()
  defp detect_linux_npu do
    # Check for various Linux NPUs (Qualcomm, Intel, etc.)
    # This is a placeholder - would need specific detection logic
    false
  end

  @spec count_vulkan_gpus() :: non_neg_integer()
  defp count_vulkan_gpus do
    # Count Vulkan physical devices
    # This would require Vulkan API calls - simplified
    if detect_vulkan(), do: 1, else: 0
  end

  @spec count_cuda_gpus() :: non_neg_integer()
  defp count_cuda_gpus do
    task =
      Task.async(fn ->
        System.cmd("nvidia-smi", ["--list-gpus"], stderr_to_stdout: true)
      end)

    case Task.yield(task, 2000) do
      {:ok, {output, 0}} ->
        String.split(output, "\n") |> Enum.count(fn line -> String.trim(line) != "" end)

      _ ->
        Task.shutdown(task, :brutal_kill)
        0
    end
  end

  @spec count_opencl_devices() :: non_neg_integer()
  defp count_opencl_devices do
    task =
      Task.async(fn ->
        System.cmd("clinfo", ["-l"], stderr_to_stdout: true)
      end)

    case Task.yield(task, 2000) do
      {:ok, {output, 0}} ->
        # Count lines that look like devices
        output
        |> String.split("\n")
        |> Enum.count(fn line ->
          String.contains?(line, "Device") or String.contains?(line, "GPU")
        end)

      _ ->
        Task.shutdown(task, :brutal_kill)
        0
    end
  end

  @spec count_linux_npus() :: non_neg_integer()
  defp count_linux_npus do
    # Placeholder for Linux NPU counting
    0
  end

  @doc """
  Logs detected hardware capabilities.
  """
  @spec log_capabilities() :: :ok
  def log_capabilities do
    caps = detect()

    Logger.info("Hardware Detection Results:")
    Logger.info("  Vulkan: #{caps.vulkan}")
    Logger.info("  CUDA: #{caps.cuda}")
    Logger.info("  Metal: #{caps.metal}")
    Logger.info("  OpenCL: #{caps.opencl}")
    Logger.info("  RT Cores: #{caps.rt_cores}")
    Logger.info("  Tensor Cores: #{caps.tensor_cores}")
    Logger.info("  NPU: #{caps.npu}")
    Logger.info("  SIMD AVX2: #{caps.simd_avx2}")
    Logger.info("  SIMD AVX-512: #{caps.simd_avx512}")
    Logger.info("  SIMD NEON: #{caps.simd_neon}")
    Logger.info("  CPU Cores: #{caps.core_count}")
    Logger.info("  GPUs: #{caps.gpu_count}")
    Logger.info("  NPUs: #{caps.npu_count}")
  end
end
