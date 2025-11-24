defmodule BrokenRecord.Zero.NIF do
  @moduledoc """
  Native Implemented Functions for physics simulation.
  """

  @on_load :load_nif

  def load_nif do
    priv_dir = :code.priv_dir(:broken_record_zero)

    # Handle case where priv_dir might be an error tuple
    priv_dir = case priv_dir do
      {:error, _} ->
        # Fallback: try to find priv directory manually
        Path.expand("./priv", File.cwd!())
      _ ->
        priv_dir
    end

    expected_nif_file = Path.join(priv_dir, "brokenrecord_physics")

    IO.inspect("DEBUG: NIF loading - priv_dir: #{inspect(priv_dir)}")
    IO.inspect("DEBUG: NIF loading - expected_nif_file: #{inspect(expected_nif_file)}")

    # Check what files actually exist in priv directory
    case File.ls("#{priv_dir}") do
      {:ok, files} ->
        IO.inspect("DEBUG: Files in priv_dir: #{inspect(files)}")
        so_files = Enum.filter(files, &String.ends_with?(&1, ".so"))
        IO.inspect("DEBUG: Found .so files: #{inspect(so_files)}")

        # Try to find the expected exact filename first
        load_result = :erlang.load_nif(expected_nif_file, 0)
        IO.inspect("DEBUG: Direct NIF load result: #{inspect(load_result)}")
        load_result
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
        case :erlang.fun_info({__MODULE__, :create_particle_system, 1}, :type) do
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

  def native_integrate(_system, _dt, _steps, _opts) do
    # This function will be replaced by the C NIF when the module loads
    # If NIF is not loaded, raise an error
    IO.puts("DEBUG: native_integrate called but NIF not loaded")
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
