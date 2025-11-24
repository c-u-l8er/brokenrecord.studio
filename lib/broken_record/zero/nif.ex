defmodule BrokenRecord.Zero.NIF do
  @moduledoc """
  Native Implemented Functions for physics simulation.
  """

  @on_load :load_nif

  def load_nif do
    priv_dir = :code.priv_dir(:broken_record_zero)
    expected_nif_file = :filename.join(priv_dir, "brokenrecord_physics")

    IO.inspect("DEBUG: NIF loading - priv_dir: #{inspect(priv_dir)}")
    IO.inspect("DEBUG: NIF loading - expected_nif_file: #{inspect(expected_nif_file)}")

    # Check what files actually exist in priv directory
    case File.ls("#{priv_dir}") do
      {:ok, files} ->
        IO.inspect("DEBUG: Files in priv_dir: #{inspect(files)}")
        so_files = Enum.filter(files, &String.ends_with?(&1, ".so"))
        IO.inspect("DEBUG: Found .so files: #{inspect(so_files)}")

        # Try to find the expected exact filename first
        result = if Enum.member?(so_files, "brokenrecord_physics.so") do
          IO.inspect("DEBUG: Found expected brokenrecord_physics.so")
          :erlang.load_nif(expected_nif_file, 0)
        else
          IO.inspect("DEBUG: Expected brokenrecord_physics.so not found, trying first available .so file")
          # Fallback: try to use the first available .so file
          case so_files do
            [first_so | _] ->
              # Remove .so extension if present to avoid double extension
              base_name = String.replace_suffix(first_so, ".so", "")
              fallback_path = :filename.join(priv_dir, base_name)
              IO.inspect("DEBUG: Trying fallback NIF file: #{inspect(fallback_path)}")
              :erlang.load_nif(fallback_path, 0)
            [] ->
              IO.inspect("DEBUG: No .so files found, NIF loading will fail")
              {:error, :no_so_files}
          end
        end

        IO.inspect("DEBUG: NIF load result: #{inspect(result)}")
        result
      {:error, reason} ->
        IO.inspect("DEBUG: Cannot list priv_dir: #{inspect(reason)}")
        {:error, reason}
    end
  end

  def create_particle_system(state) do
    IO.inspect("DEBUG: create_particle_system - checking function availability")
    IO.inspect("DEBUG: create_particle_system - module: #{__MODULE__}")
    IO.inspect("DEBUG: create_particle_system - functions: #{:erlang.apply(__MODULE__, :module_info, [:functions])}")
    IO.inspect("DEBUG: create_particle_system - state type: #{inspect(state)}")
    IO.inspect("DEBUG: create_particle_system - state keys: #{inspect(Map.keys(state))}")

    case function_exported?(__MODULE__, :create_particle_system, 1) do
      true ->
        IO.inspect("DEBUG: create_particle_system - function is exported, calling NIF")
        IO.inspect("DEBUG: create_particle_system - about to call with state: #{inspect(state, pretty: true)}")

        # Check if the function is actually the NIF or a fallback
        case :erlang.fun_info(__MODULE__.create_particle_system, :type) do
          {:type, :external} ->
            IO.inspect("DEBUG: create_particle_system - confirmed NIF function (external)")
          {:type, :local} ->
            IO.inspect("DEBUG: create_particle_system - this is a local Elixir function, not NIF")
          other ->
            IO.inspect("DEBUG: create_particle_system - unknown function type: #{inspect(other)}")
        end

        result = :erlang.apply(__MODULE__, :create_particle_system, [state])
        IO.inspect("DEBUG: create_particle_system - NIF result: #{inspect(result)}")
        result
      false ->
        IO.inspect("DEBUG: create_particle_system - function NOT exported, checking loaded modules")
        IO.inspect("DEBUG: create_particle_system - loaded modules: #{:code.all_loaded() |> Enum.filter(fn {mod, _} -> mod == __MODULE__ end)}")
        IO.warn("NIF create_particle_system not available, falling back to interpreted")
        # Fallback to interpreted for now
        {:error, :nif_not_loaded}
    end
  end

  def native_integrate(system, dt, steps, opts) do
    # This function will be replaced by the C NIF when the module loads
    # If NIF is not loaded, raise an error
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_elixir_state(system) do
    case function_exported?(__MODULE__, :to_elixir_state, 1) do
      true -> :erlang.apply(__MODULE__, :to_elixir_state, [system])
      false ->
        IO.warn("NIF to_elixir_state not available")
        system
    end
  end
end
