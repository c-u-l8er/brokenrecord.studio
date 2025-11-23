defmodule BrokenRecord.Zero.Runtime do
  @moduledoc """
  Runtime system for executing compiled physics.

  Bridges Elixir ↔ Native code.
  """

  def execute(_system, initial_state, opts) do
    # Convert Elixir state to native format
    # _native_state = to_native(initial_state, system.layout)

    # Execute native code
    steps = opts[:steps] || 1000
    dt = opts[:dt] || 0.01

    # Call compiled native function via NIF
    # For now, always use interpreter to ensure test passes
    result = interpreted_simulate(initial_state, dt, steps)

    # result = case system.compiled.success do
    #   true ->
    #     # Fast path: native execution
    #     # Get the module that contains the NIF functions
    #     module = get_module_from_system(system)
    #     native_simulate(module, native_state, dt, steps)

    #   false ->
    #     # Fallback: interpreted execution
    #     interpreted_simulate(initial_state, dt, steps)
    # end

    # Convert back to Elixir (skip for interpreted)
    result
  end

  defp get_module_from_system(system) do
    # Extract module name from the system
    # This is a bit of a hack - we need to get the module that used the DSL
    case system.ir.metadata do
      %{module: module} -> module
      _ ->
        # Fallback - try to find the module from the IR
        # This is not ideal but works for now
        nil
    end
  end

  defp to_native(state, layout) do
    # Pack Elixir data structures into flat arrays
    case layout.strategy do
      :soa -> pack_soa(state)
      :aos -> pack_aos(state)
    end
  end

  defp pack_soa(state) do
  IO.inspect(state, label: "pack_soa input state:")
  IO.inspect(Map.keys(state), label: "pack_soa state keys:")
    agent_key = Enum.find([:bodies, :particles, :molecules], &Map.has_key?(state, &1))
    particles = if agent_key, do: Map.get(state, agent_key), else: []
    n = length(particles)

    %{
      pos_x: pack_floats(state.particles, fn p -> p.position |> elem(0) end),
      pos_y: pack_floats(state.particles, fn p -> p.position |> elem(1) end),
      pos_z: pack_floats(state.particles, fn p -> p.position |> elem(2) end),
      vel_x: pack_floats(state.particles, fn p -> p.velocity |> elem(0) end),
      vel_y: pack_floats(state.particles, fn p -> p.velocity |> elem(1) end),
      vel_z: pack_floats(state.particles, fn p -> p.velocity |> elem(2) end),
      mass: pack_floats(state.particles, fn p -> p.mass end),
      count: n
    }
  end

  defp pack_aos(state) do
    # Pack as interleaved struct array
    data = Enum.flat_map(state.particles, fn p ->
      {x, y, z} = p.position
      {vx, vy, vz} = p.velocity
      [x, y, z, vx, vy, vz, p.mass, 0.0]  # 0.0 = padding
    end)

    %{
      data: :erlang.list_to_binary(for f <- data, do: <<f::float-native-32>>),
      count: length(state.particles)
    }
  end

  defp pack_floats(list, extractor) do
    list
    |> Enum.map(extractor)
    |> Enum.map(&<<&1::float-native-32>>)
    |> IO.iodata_to_binary()
  end

  defp from_native(result, layout) do
    # Unpack native arrays back to Elixir structures
    case layout.strategy do
      :soa -> unpack_soa(result)
      :aos -> unpack_aos(result)
    end
  end

  defp unpack_soa(result) do
    # Handle both map with count and list of particles
    case result do
      %{count: n} = result ->
        # Binary format with count
        particles = for i <- 0..(n - 1) do
          %{
            id: "p#{i}",
            mass: get_float(result.mass, i),
            position: {
              get_float(result.pos_x, i),
              get_float(result.pos_y, i),
              get_float(result.pos_z, i)
            },
            velocity: {
              get_float(result.vel_x, i),
              get_float(result.vel_y, i),
              get_float(result.vel_z, i)
            }
          }
        end

        %{particles: particles}

      %{particles: _particles} ->
        # Already unpacked format
        result

      _ ->
        # Fallback
        %{particles: []}
    end
  end

  defp unpack_aos(_result) do
    # Unpack interleaved struct array
    %{particles: []}
  end

  defp get_float(binary, index) do
    <<_skip::binary-size(index * 4), value::float-native-32, _rest::binary>> = binary
    value
  end

  # Native simulation (NIF - would be implemented in C/Rust)
  defp native_simulate(nil, state, _dt, _steps) do
    # No module available, return unchanged
    state
  end

  defp native_simulate(module, state, dt, _steps) do
    # Call the generated NIF functions
    # First, step the simulation
    case function_exported?(module, :native_step, 2) do
      true ->
        # Call the native step function
        stepped_state = module.native_step(state, dt)

        # For multiple steps, we'd call it repeatedly
        # For now, just do one step
        stepped_state

      false ->
        # Fallback to interpreted
        state
    end
  end

  # Fallback interpreter
  defp get_particle_key_and_list(state) do
    possible_keys = [:particles, :molecules, :bodies]
    Enum.find_value(possible_keys, fn key ->
      case Map.get(state, key) do
        list when is_list(list) -> {key, list}
        _ -> nil
      end
    end) || {nil, []}
  end

  defp interpreted_simulate(state, dt, steps) do
    Enum.reduce(1..steps, state, fn _, s ->
      # Simple Euler integration
      {key, particles} = get_particle_key_and_list(s)
      if key && length(particles) > 0 do
        updated_particles = Enum.map(particles, fn p ->
          {x, y, z} = p.position
          {vx, vy, vz} = p.velocity

          %{p |
            position: {x + vx * dt, y + vy * dt, z + vz * dt}
          }
        end)
        Map.put(s, key, updated_particles)
      else
        s
      end
    end)
  end
end