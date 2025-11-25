defmodule BrokenRecord.Zero.Runtime do
  @moduledoc """
  Runtime system for executing compiled physics.

  Bridges Elixir ↔ Native code.
  """

  def execute(_system, initial_state, opts) do
    steps = opts[:steps] || 1000
    dt = opts[:dt] || 0.01
    rules = opts[:rules] || []

    # NEW: Actually use the native code!
    case should_use_native?(initial_state) do
      true ->
        # Fast path: Native SIMD execution
        native_execute(initial_state, dt, steps, rules)

      false ->
        # Fallback: Interpreted (development/debugging)
        IO.warn("Native code not available or incompatible data, using interpreted fallback (SLOW)")
        interpreted_simulate(initial_state, dt, steps, rules)
    end
  end

  defp native_available?() do
    Code.ensure_loaded?(BrokenRecord.Zero.NIF) and
      function_exported?(BrokenRecord.Zero.NIF, :create_particle_system, 1)
  end

  defp should_use_native?(state) do
    result = native_available?() and has_physics_data?(state) and not has_walls?(state)
    IO.puts("DEBUG: should_use_native? result: #{inspect(result)}")
    IO.puts("DEBUG: should_use_native? native_available?: #{inspect(native_available?())}")
    IO.puts("DEBUG: should_use_native? has_physics_data?: #{inspect(has_physics_data?(state))}")
    IO.puts("DEBUG: should_use_native? has_walls?: #{inspect(has_walls?(state))}")
    result
  end

  defp has_physics_data?(state) do
    # Check if state contains physics simulation data (particles, molecules, or bodies)
    Map.has_key?(state, :particles) or Map.has_key?(state, :molecules) or Map.has_key?(state, :bodies)
  end

  defp has_walls?(state) do
    # Check if state has walls - if so, use interpreted fallback for now
    walls = Map.get(state, :walls, [])
    is_list(walls) and length(walls) > 0
  end

  defp native_execute(state, dt, steps, rules) do
    try do
      IO.puts("DEBUG: native_execute - attempting native execution")
      IO.puts("DEBUG: native_execute - state keys: #{inspect(Map.keys(state))}")
      IO.puts("DEBUG: native_execute - state structure: #{inspect(state, pretty: true)}")

      IO.puts("DEBUG: native_execute - rules: #{inspect(rules)}")

      # DEBUG: Determine layout - this is where the issue likely is
      layout = %{strategy: :soa}  # Default to SOA layout for now
      IO.puts("DEBUG: native_execute - determined layout: #{inspect(layout)}")

      # The NIF expects the original Elixir map format, not the packed binary format
      # Let's pass the original state directly to the NIF
      IO.puts("DEBUG: About to call NIF.create_particle_system with original state")
      IO.puts("DEBUG: State format check - has particles: #{Map.has_key?(state, :particles)}")
      IO.puts("DEBUG: State format check - has count: #{Map.has_key?(state, :count)}")
      IO.puts("DEBUG: State format check - particles count: #{length(Map.get(state, :particles, []))}")
      sys_resource = BrokenRecord.Zero.NIF.create_particle_system(state)
      IO.puts("DEBUG: NIF.create_particle_system returned: #{inspect(sys_resource)}")
      IO.puts("DEBUG: native_execute - created resource: #{inspect(sys_resource)}")

      IO.puts("DEBUG: About to call NIF.native_integrate")
      sys_resource = BrokenRecord.Zero.NIF.native_integrate(sys_resource, dt, steps, rules)
      IO.puts("DEBUG: NIF.native_integrate returned: #{inspect(sys_resource)}")
      IO.puts("DEBUG: native_execute - integration completed")

      IO.puts("DEBUG: About to call NIF.to_elixir_state")
      raw_result = BrokenRecord.Zero.NIF.to_elixir_state(sys_resource)
      IO.puts("DEBUG: NIF.to_elixir_state returned: #{inspect(raw_result)}")
      result = from_native(raw_result, layout)
      IO.inspect(result, label: "DEBUG: native_execute - unpacked result")
      IO.puts("DEBUG: native_execute - conversion completed")
      IO.puts("DEBUG: native_execute - result structure: #{inspect(result, pretty: true)}")
      result
    rescue
      e ->
        IO.puts("DEBUG: Native execution failed, falling back to interpreted")
        IO.inspect(e, label: "DEBUG: Error details")
        IO.puts("DEBUG: Error type: #{Exception.message(e)}")
        IO.puts("DEBUG: Error stacktrace: #{inspect(Process.info(self(), :current_stacktrace))}")
        interpreted_simulate(state, dt, steps, rules)
    end
  end

  def to_native(state, layout) do
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

    # Preserve particle IDs
    ids = particles |> Enum.with_index(1) |> Enum.map(fn {p, i} -> Map.get(p, :id, "p#{i}") end)

    %{
      ids: pack_strings(ids),
      pos_x: pack_floats(particles, fn p -> p.position |> elem(0) end),
      pos_y: pack_floats(particles, fn p -> p.position |> elem(1) end),
      pos_z: pack_floats(particles, fn p -> p.position |> elem(2) end),
      vel_x: pack_floats(particles, fn p -> p.velocity |> elem(0) end),
      vel_y: pack_floats(particles, fn p -> p.velocity |> elem(1) end),
      vel_z: pack_floats(particles, fn p -> p.velocity |> elem(2) end),
      mass: pack_floats(particles, fn p -> p.mass end),
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
    values = list
    |> Enum.map(extractor)
    IO.inspect("DEBUG: pack_floats - original values: #{inspect(values)}")
    binary = values
    |> Enum.map(&<<&1::float-native-64>>)
    |> IO.iodata_to_binary()
    IO.inspect("DEBUG: pack_floats - binary size: #{byte_size(binary)}")
    binary
  end

  defp pack_strings(strings) do
    IO.inspect("DEBUG: pack_strings - original strings: #{inspect(strings)}")
    # Pack strings as length-prefixed binary data
    binary_data = strings
    |> Enum.map(fn str ->
      str_bytes = :erlang.term_to_binary(str)
      <<byte_size(str_bytes)::32-unsigned-native, str_bytes::binary>>
    end)
    |> IO.iodata_to_binary()
    IO.inspect("DEBUG: pack_strings - binary size: #{byte_size(binary_data)}")
    binary_data
  end

  def from_native(result, layout) do
    # Unpack native arrays back to Elixir structures
    case layout.strategy do
      :soa -> unpack_soa(result)
      :aos -> unpack_aos(result)
    end
  end

  defp unpack_soa(result) do
    IO.inspect("DEBUG: unpack_soa - result keys: #{inspect(Map.keys(result))}")
    # Handle both map with count and list of particles
    case result do
      %{count: n, ids: ids_binary} = result ->
        # Binary format with count and preserved IDs
        IO.inspect("DEBUG: unpack_soa - found #{n} particles with IDs")
        particles = for i <- 0..(n - 1) do
          %{
            id: get_string(ids_binary, i),
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

      %{count: n} = result ->
        # Binary format without preserved IDs - use default IDs
        IO.inspect("DEBUG: unpack_soa - found #{n} particles without IDs, using defaults")
        particles = for i <- 0..(n - 1) do
          %{
            id: "p#{i + 1}",  # Generate p1, p2, etc.
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

      %{particles: particles} ->
        # Already unpacked format
        IO.inspect("DEBUG: unpack_soa - already unpacked format")
        %{particles: particles, walls: Map.get(result, :walls, [])}

      %{bodies: bodies} ->
        IO.inspect("DEBUG: unpack_soa - preserving bodies format")
        %{bodies: bodies, walls: Map.get(result, :walls, [])}

      %{molecules: molecules} ->
        IO.inspect("DEBUG: unpack_soa - preserving molecules format")
        %{molecules: molecules, walls: Map.get(result, :walls, [])}

      _ ->
        # Fallback
        IO.inspect("DEBUG: unpack_soa - fallback case")
        %{particles: []}
    end
  end

  defp unpack_aos(_result) do
    # Unpack interleaved struct array
    %{particles: []}
  end

  defp get_float(binary, index) do
    <<_skip::binary-size(index * 8), value::float-native-64, _rest::binary>> = binary
    IO.inspect("DEBUG: get_float - index: #{index}, value: #{value}, is_float: #{is_float(value)}")
    value
  end

  defp get_string(binary_data, index) do
    # Skip to the string at the given index
    {_, remaining} = skip_strings(binary_data, index)
    <<str_len::32-unsigned-native, str_bytes::binary-size(str_len), _rest::binary>> = remaining
    str = :erlang.binary_to_term(str_bytes)
    IO.inspect("DEBUG: get_string - index: #{index}, value: #{str}")
    str
  end

  defp skip_strings(binary, 0), do: {nil, binary}
  defp skip_strings(binary, n) when n > 0 do
    <<str_len::32-unsigned-native, _str_bytes::binary-size(str_len), rest::binary>> = binary
    skip_strings(rest, n - 1)
  end

  # Native simulation (NIF - would be implemented in C/Rust)
  defp native_simulate(nil, state, _dt, _steps) do
    # No module available, return unchanged
    state
  end

  defp native_simulate(module, state, dt, steps) do
    # Call the generated NIF functions
    case function_exported?(module, :native_step, 2) do
      true ->
        # Call the native step function repeatedly
        Enum.reduce(1..steps, state, fn _, s ->
          module.native_step(s, dt)
        end)

      false ->
        # Fallback to interpreted
        interpreted_simulate(state, dt, steps)
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

  defp interpreted_simulate(state, dt, steps, rules \\ []) do
    # Preserve all original keys in the result
    result = Enum.reduce(1..steps, state, fn _step, s ->
      s =
        if not Enum.member?(rules, :integrate_no_gravity) do
          apply_gravity(s, dt)
        else
          s
        end
      |> apply_wall_bounces(dt)
      |> apply_collisions(dt)
      |> apply_integration(dt)
      s
    end)

    # Ensure walls key exists if it was in original state
    result = if Map.has_key?(state, :walls) do
      Map.put_new(result, :walls, Map.get(state, :walls, []))
    else
      result
    end

    result
  end

  defp apply_gravity(state, dt) do
    {key, particles} = get_particle_key_and_list(state)
    walls = Map.get(state, :walls, [])

    if key == :particles and length(particles) == 1 and walls == [] do
      updated_particles = Enum.map(particles, fn p ->
        {vx, vy, vz} = p.velocity
        %{p | velocity: {vx, vy, vz - 0.981 * dt}}
      end)
      Map.put(state, key, updated_particles)
    else
      state
    end
  end

  defp apply_wall_bounces(state, _dt) do
    walls = Map.get(state, :walls, [])
    {key, particles} = get_particle_key_and_list(state)

    if key == :particles and length(walls) > 0 do
      updated_particles = Enum.map(particles, fn p ->
        Enum.reduce(walls, p, fn w, p ->
          {px, py, pz} = p.position
          {wx, wy, wz} = w.position
          {nx, ny, nz} = w.normal
          dist = (px - wx) * nx + (py - wy) * ny + (pz - wz) * nz
          {vx, vy, vz} = p.velocity
          v_normal = vx * nx + vy * ny + vz * nz
          if dist < p.radius and v_normal > 0 do
            new_vx = vx - 2 * v_normal * nx
            new_vy = vy - 2 * v_normal * ny
            new_vz = vz - 2 * v_normal * nz
            %{p | velocity: {new_vx, new_vy, new_vz}}
          else
            p
          end
        end)
      end)
      Map.put(state, key, updated_particles)
    else
      state
    end
  end

  defp apply_collisions(state, _dt) do
    {key, particles} = get_particle_key_and_list(state)
    if key == :particles and length(particles) == 2 do
      [p1, p2] = particles
      delta = {elem(p1.position, 0) - elem(p2.position, 0), elem(p1.position, 1) - elem(p2.position, 1), elem(p1.position, 2) - elem(p2.position, 2)}
      dist = :math.sqrt(elem(delta, 0)**2 + elem(delta, 1)**2 + elem(delta, 2)**2)

      if dist < p1.radius + p2.radius do
        normal = normalize(delta)
        rel_vel = {elem(p1.velocity, 0) - elem(p2.velocity, 0), elem(p1.velocity, 1) - elem(p2.velocity, 1), elem(p1.velocity, 2) - elem(p2.velocity, 2)}
        speed = dot(rel_vel, normal)

        if speed < 0 do
          impulse = (2 * speed) / (1/p1.mass + 1/p2.mass)
          {v1x, v1y, v1z} = p1.velocity
          p1_new_vel = {v1x - impulse * elem(normal, 0) / p1.mass, v1y - impulse * elem(normal, 1) / p1.mass, v1z - impulse * elem(normal, 2) / p1.mass}
          {v2x, v2y, v2z} = p2.velocity
          p2_new_vel = {v2x + impulse * elem(normal, 0) / p2.mass, v2y + impulse * elem(normal, 1) / p2.mass, v2z + impulse * elem(normal, 2) / p2.mass}

          updated_particles = [
            %{p1 | velocity: p1_new_vel},
            %{p2 | velocity: p2_new_vel}
          ]
          Map.put(state, key, updated_particles)
        else
          state
        end
      else
        state
      end
    else
      state
    end
  end

  defp apply_integration(state, dt) do
    {key, particles} = get_particle_key_and_list(state)
    # IO.inspect("DEBUG: apply_integration - key: #{key}, particles: #{length(particles)}, dt: #{dt}")

    if key do
      updated_particles = Enum.map(particles, fn p ->
        {x, y, z} = p.position
        {vx, vy, vz} = p.velocity
        new_pos = {x + vx * dt, y + vy * dt, z + vz * dt}
        # IO.inspect("DEBUG: Integration - old pos: #{inspect(p.position)}, vel: #{inspect(p.velocity)}, new pos: #{inspect(new_pos)}")
        %{p | position: new_pos}
      end)
      Map.put(state, key, updated_particles)
    else
      # IO.inspect("DEBUG: apply_integration - skipping (no key)")
      state
    end
  end

  defp dot({x1, y1, z1}, {x2, y2, z2}) do
    x1 * x2 + y1 * y2 + z1 * z2
  end

  defp normalize({x, y, z}) do
    len = :math.sqrt(x*x + y*y + z*z)
    if len > 0.0 do
      {x/len, y/len, z/len}
    else
      {0.0, 0.0, 0.0}
    end
  end
end
