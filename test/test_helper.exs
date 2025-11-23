ExUnit.start()

# Compile and load examples that tests depend on
Code.require_file("examples/gravity_simulation.ex")
Code.require_file("examples/actor_model.ex")
Code.require_file("examples/chemical_reaction_net.ex")
Code.require_file("examples/my_physics.ex")

# Load subdirectory test helpers
Code.require_file("test/examples/test_helper.exs")

# Configure ExUnit for better test output
ExUnit.configure(exclude: :benchmark, max_failures: 1)
