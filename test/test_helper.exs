ExUnit.start()

# Load subdirectory test helpers
Code.require_file("test/examples/test_helper.exs")

# Configure ExUnit for better test output
ExUnit.configure(exclude: :benchmark, max_failures: 1)
