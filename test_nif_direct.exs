# Test NIF directly without mix
Code.require_file("lib/broken_record/zero/nif.ex")

# Test basic NIF loading
IO.puts("Testing NIF loading...")

# Test create_particle_system
state = %{
  particles: [
    %{
      id: "test1",
      position: {0.0, 0.0, 0.0},
      velocity: {0.0, 0.0, 0.0},
      mass: 1.0,
      radius: 1.0
    }
  ]
}

IO.puts("About to call create_particle_system...")
result = BrokenRecord.Zero.NIF.create_particle_system(state)
IO.puts("create_particle_system result: #{inspect(result)}")
