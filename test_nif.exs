#!/usr/bin/env elixir

# Simple test script to verify NIF loading without mix compilation issues

IO.puts("Testing NIF loading...")

# Try to load the NIF module directly
try do
  Code.require_file("lib/broken_record/zero/nif.ex")
  IO.puts("✓ NIF module loaded successfully")

  # Test if NIF functions are available
  if function_exported?(BrokenRecord.Zero.NIF, :create_particle_system, 1) do
    IO.puts("✓ create_particle_system function is available")
  else
    IO.puts("✗ create_particle_system function is NOT available")
  end

  # Try to create a simple test state
  test_state = %{
    particles: [%{
      id: "test_particle",
      position: {0.0, 0.0, 10.0},
      velocity: {0.0, 0.0, 0.0},
      mass: 1.0,
      radius: 1.0
    }],
    walls: []
  }

  IO.puts("✓ Test state created")
  IO.puts("State: #{inspect(test_state)}")

  # Try to call the NIF
  result = BrokenRecord.Zero.NIF.create_particle_system(test_state)
  IO.puts("✓ NIF create_particle_system called successfully")
  IO.puts("Result: #{inspect(result)}")

rescue
  e ->
    IO.puts("✗ Error loading or calling NIF: #{inspect(e)}")
    IO.puts("Error type: #{Exception.message(e)}")
end
