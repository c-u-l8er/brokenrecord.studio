#!/usr/bin/env elixir

# Simple web server to run BrokenRecord.Zero examples and capture output
defmodule ExampleServer do
  use Plug.Router

  plug Plug.Logger
  plug :match
  plug :dispatch

  get "/" do
    send_resp(conn, 200, "Example Server Running")
  end

  post "/run/:example" do
    example_name = example_params["example"]

    # Load and compile the example
    case load_and_run_example(example_name) do
      {:ok, output} ->
        send_resp(conn, 200, Jason.encode!(%{success: true, output: output}))

      {:error, error} ->
        send_resp(conn, 500, Jason.encode!(%{success: false, error: error}))
    end
  end

  match _ do
    send_resp(conn, 404, "Not found")
  end

  defp load_and_run_example(example_name) do
    try do
      # Capture IO output
      {:ok, output} = ExUnit.CaptureIO.capture_io(fn ->
        # Load the example module
        module_name = "Examples." <> String.capitalize(String.replace(example_name, ".ex", "")) |> String.replace("_", "")
        module = Module.concat([Examples, String.capitalize(String.replace(example_name, ".ex", "")) |> String.replace("_", "")])

        # Ensure the module is loaded and compiled
        Code.require_file("examples/#{example_name}", __DIR__)

        # Run the example with a simple simulation
        case module do
          Examples.ActorModel ->
            state = Examples.ActorModel.actor_system()
            result = Examples.ActorModel.simulate(state, steps: 10, dt: 0.1)
            IO.puts("Actor Model Simulation Results:")
            IO.puts("Final state: #{inspect(result, pretty: true)}")
            stats = Examples.ActorModel.system_stats(result)
            IO.puts("System stats: #{inspect(stats, pretty: true)}")

          Examples.ChemicalReactionNet ->
            state = Examples.ChemicalReactionNet.chemical_mixture(10, 10, 5, 5)
            result = Examples.ChemicalReactionNet.simulate(state, steps: 10, dt: 0.1)
            IO.puts("Chemical Reaction Network Results:")
            IO.puts("Initial molecule counts: #{inspect(Examples.ChemicalReactionNet.count_molecules(state), pretty: true)}")
            IO.puts("Final molecule counts: #{inspect(Examples.ChemicalReactionNet.count_molecules(result), pretty: true)}")
            IO.puts("Total mass conserved: #{Examples.ChemicalReactionNet.total_mass(state) == Examples.ChemicalReactionNet.total_mass(result)}")

          Examples.GravitySimulation ->
            state = Examples.GravitySimulation.solar_system()
            result = Examples.GravitySimulation.simulate(state, steps: 10, dt: 0.1)
            IO.puts("Gravity Simulation Results:")
            IO.puts("Initial energy: #{Examples.GravitySimulation.total_energy(state)}")
            IO.puts("Final energy: #{Examples.GravitySimulation.total_energy(result)}")
            conservation = Examples.GravitySimulation.verify_conservation(state, result)
            IO.puts("Conservation verified: #{inspect(conservation, pretty: true)}")

          Examples.MyPhysics ->
            # This one is a bit different - it's a demo module
            Code.require_file("examples/my_physics.ex", __DIR__)
            IO.puts("MyPhysics Demo:")
            IO.puts("This example defines a custom physics system with GPU acceleration support.")
            IO.puts("Key features:")
            IO.puts("  - Particle-particle collisions with momentum conservation")
            IO.puts("  - Wall bouncing with reflection")
            IO.puts("  - Gravity integration")
            IO.puts("  - CUDA compilation target for GPU acceleration")
            IO.puts("  - Spatial hashing and SIMD optimization")

          _ ->
            IO.puts("Unknown example: #{example_name}")
        end
      end)

      {:ok, output}
    rescue
      error ->
        {:error, Exception.format(:error, error, __STACKTRACE__)}
    end
  end
end

defmodule ExampleServer.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      {Plug.Cowboy, scheme: :http, plug: ExampleServer, options: [port: 4001]}
    ]

    Supervisor.init([strategy: :one_for_one, children: children])
  end
end

# Start the server
{:ok, _} = ExampleServer.Supervisor.start_link([])

IO.puts("Example server running on http://localhost:4001")
IO.puts("Press Ctrl+C to stop")

# Keep the server running
Process.sleep(:infinity)
