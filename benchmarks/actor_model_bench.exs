defmodule ActorModelBench do
  @moduledoc """
  Benchmarks for the Actor Model runtime performance using BrokenRecord Zero DSL.
  Measures simulation performance for different system sizes and workloads.
  """

  alias Examples.ActorModel
  Code.require_file("examples/actor_model.ex")

  def run do
    IO.puts("Running Actor Model Runtime Benchmarks...")

    Benchee.run(
      %{
        "Actor System - Small (4 actors, 100 steps)" => fn ->
          ActorModel.run_simulation(ActorModel.actor_system(), steps: 100, dt: 0.01)
        end,
        "Actor System - Small (4 actors, 1000 steps)" => fn ->
          ActorModel.run_simulation(ActorModel.actor_system(), steps: 1000, dt: 0.01)
        end,
        "Actor System - Medium (100 actors, 100 steps)" => fn ->
          medium_system = create_medium_system()
          ActorModel.ActorRuntime.simulate(medium_system, steps: 100, dt: 0.01)
        end,
        "Actor System - Large (1000 actors, 100 steps)" => fn ->
          large_system = create_large_system()
          ActorModel.ActorRuntime.simulate(large_system, steps: 100, dt: 0.01)
        end,
        "Message Throughput (10k messages)" => fn ->
          system = ActorModel.actor_system()
          send_bulk_messages(system, 10000)
        end,
        "Actor Creation (1000 actors)" => fn ->
          create_dynamic_actors(1000)
        end
      },
      memory_time: 2,
      time: 5,
      warmup: 2,
      print: [configuration: false],
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true},
        {Benchee.Formatters.HTML, file: "benchmarks/actor_model_benchmarks.html"}
      ]
    )

    IO.puts("Actor Model benchmarks complete!")
  end

  defp create_medium_system do
    num_actors = 100
    actors = for i <- 1..num_actors, do: %{
      pid: i,
      state: %{count: 0},
      mailbox: [],
      behavior: :counter,
      supervisor: 0,
      status: :running,
      processing_time: 0.0
    }
    messages = for i <- 1..(num_actors * 10), do: %{
      sender: 9999,
      receiver: rem(i-1, num_actors) + 1,
      content: nil,
      type: :increment,
      timestamp: 0.0
    }
    supervisors = [%{
      pid: 0,
      children: (for i <- 1..num_actors, do: i),
      strategy: :one_for_one,
      restart_policy: :permanent,
      max_restarts: 5,
      restart_count: 0
    }]
    scheduler = %{
      ready_queue: [],
      running_actors: (for i <- 1..10, do: i),
      time_slice: 1.0,
      load_balance_strategy: :round_robin
    }
    %{actors: actors, messages: messages, supervisors: supervisors, scheduler: scheduler}
  end

  defp create_large_system do
    num_actors = 1000
    actors = for i <- 1..num_actors, do: %{
      pid: i,
      state: %{count: 0},
      mailbox: [],
      behavior: :counter,
      supervisor: 0,
      status: :running,
      processing_time: 0.0
    }
    messages = for i <- 1..(num_actors * 10), do: %{
      sender: 9999,
      receiver: rem(i-1, num_actors) + 1,
      content: nil,
      type: :increment,
      timestamp: 0.0
    }
    supervisors = [%{
      pid: 0,
      children: (for i <- 1..num_actors, do: i),
      strategy: :one_for_one,
      restart_policy: :permanent,
      max_restarts: 5,
      restart_count: 0
    }]
    scheduler = %{
      ready_queue: [],
      running_actors: (for i <- 1..min(100, num_actors), do: i),
      time_slice: 1.0,
      load_balance_strategy: :round_robin
    }
    %{actors: actors, messages: messages, supervisors: supervisors, scheduler: scheduler}
  end

  defp send_bulk_messages(system, count) do
    Enum.reduce(1..count, system, fn i, acc ->
      ActorModel.send_message(acc, 9999, rem(i, 4) + 1, nil, :increment)
    end)
    |> ActorModel.ActorRuntime.simulate(steps: 10, dt: 0.01)
  end

  defp create_dynamic_actors(count) do
    initial_system = ActorModel.actor_system()
    Enum.reduce(1..count, initial_system, fn _, acc ->
      # Simulate dynamic actor creation via rules
      ActorModel.ActorRuntime.simulate(acc, steps: 1, dt: 0.01)
    end)
  end
end

# Run the benchmarks
ActorModelBench.run()
