#!/usr/bin/env elixir

# Simple test to check NIF function availability
defmodule TestNIF do
  def test do
    IO.inspect("=== NIF Function Availability Test ===")

    # Check if module is loaded
    IO.inspect("Code.ensure_loaded?: #{Code.ensure_loaded?(BrokenRecord.Zero.NIF)}")

    # Check module info
    if Code.ensure_loaded?(BrokenRecord.Zero.NIF) do
      IO.inspect("Module info: #{inspect(:erlang.apply(BrokenRecord.Zero.NIF, :module_info, [:functions]))}")
    end

    # Check specific functions
    functions = [
      {:create_particle_system, 1},
      {:native_integrate, 3},
      {:to_elixir_state, 1}
    ]

    Enum.each(functions, fn {name, arity} ->
      available = function_exported?(BrokenRecord.Zero.NIF, name, arity)
      IO.inspect("Function #{name}/#{arity}: #{available}")
    end)

    # Try to call a function directly
    test_state = %{particles: [%{position: {1.0, 2.0, 3.0}, velocity: {0.1, 0.2, 0.3}, mass: 1.0}]}

    try do
      result = BrokenRecord.Zero.NIF.create_particle_system(test_state)
      IO.inspect("Direct call result: #{inspect(result)}")
    rescue
      e -> IO.inspect("Direct call error: #{Exception.message(e)}")
    end
  end
end

TestNIF.test()
