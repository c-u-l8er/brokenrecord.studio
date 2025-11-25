defmodule ActorModelComparisonBench do
  @moduledoc """
  Comprehensive benchmark comparing interpreted vs NIF performance for actor models.

  This benchmark validates execution paths and provides comparison data
  for future GPU/Vulkan implementations.
  """

  alias Examples.ActorModel
  Code.require_file("examples/actor_model.ex")

  def run do
    IO.puts("=== Actor Model Performance Comparison Benchmark ===")
    IO.puts("Testing both interpreted and NIF execution paths...")

    # First, validate NIF availability
    nif_available = check_nif_availability()
    IO.puts("NIF Available: #{nif_available}")

    # Test systems of different sizes
    test_scenarios = [
      {"Small System", 4, 100},
      {"Medium System", 50, 100},
      {"Large System", 200, 100},
      {"Stress Test", 500, 50}
    ]

    results = Enum.map(test_scenarios, fn {name, actor_count, steps} ->
      IO.puts("\n--- Testing #{name} (#{actor_count} actors, #{steps} steps) ---")

      # Create test system
      system = create_test_system(actor_count)

      # Benchmark interpreted mode
      interpreted_result = benchmark_execution_mode(system, steps, :interpreted, name)

      # Benchmark NIF mode (if available)
      nif_result = if nif_available do
        benchmark_execution_mode(system, steps, :nif, name)
      else
        %{time: nil, memory: nil, status: "NIF unavailable"}
      end

      %{
        name: name,
        actor_count: actor_count,
        steps: steps,
        interpreted: interpreted_result,
        nif: nif_result
      }
    end)

    # Generate comparison report
    generate_comparison_report(results)

    # Run Benchee benchmarks for detailed metrics
    run_detailed_benchmarks()

    IO.puts("\n=== Benchmark Complete ===")
  end

  defp check_nif_availability do
    try do
      # Check if NIF module loads and has required functions
      Code.ensure_loaded?(BrokenRecord.Zero.NIF) and
      function_exported?(BrokenRecord.Zero.NIF, :create_particle_system, 1) and
      function_exported?(BrokenRecord.Zero.NIF, :native_integrate, 4)
    rescue
      _ -> false
    end
  end

  defp create_test_system(actor_count) do
    # Create actors with different behaviors for realistic testing
    actors = for i <- 1..actor_count do
      behavior = case rem(i, 4) do
        0 -> :counter
        1 -> :calculator
        2 -> :logger
        3 -> :worker
      end

      %{
        pid: i,
        state: case behavior do
          :counter -> %{count: 0}
          :calculator -> %{last_result: 0, operations: 0}
          :logger -> %{logs: []}
          :worker -> %{completed: 0}
        end,
        mailbox: [],
        behavior: behavior,
        supervisor: 0,
        status: :running,
        processing_time: 0.0
      }
    end

    # Create root supervisor
    root_supervisor = %{
      pid: 0,
      children: Enum.map(actors, & &1.pid),
      strategy: :one_for_one,
      restart_policy: :permanent,
      max_restarts: 3,
      restart_count: 0
    }

    # Create scheduler
    scheduler = %{
      ready_queue: [],
      running_actors: Enum.take(Enum.map(actors, & &1.pid), min(10, actor_count)),
      time_slice: 1.0,
      load_balance_strategy: :round_robin
    }

    # Create initial messages (some actors start with work)
    messages = for i <- 1..div(actor_count, 2) do
      target_actor = Enum.random(actors)
      %{
        sender: 9999,  # External sender
        receiver: target_actor.pid,
        content: case target_actor.behavior do
          :counter -> nil
          :calculator -> {Enum.random(1..10), Enum.random(1..10)}
          :logger -> "Initial message #{i}"
          :worker -> %{task: "compute", data: Enum.random(1..100)}
        end,
        type: case target_actor.behavior do
          :counter -> :increment
          :calculator -> :add
          :logger -> :log
          :worker -> :work
        end,
        timestamp: :erlang.system_time(:millisecond) / 1000.0
      }
    end

    %{
      actors: actors,
      messages: messages,
      supervisors: [root_supervisor],
      scheduler: scheduler
    }
  end

  defp benchmark_execution_mode(system, steps, mode, scenario_name) do
    IO.puts("  Testing #{mode} mode...")

    # Force execution mode by manipulating runtime checks
    modified_system = case mode do
      :interpreted ->
        # Add walls to force interpreted mode (has_walls? check)
        Map.put(system, :walls, [%{position: {0, 0, 0}, normal: {0, 1, 0}}])
      :nif ->
        # Ensure no walls and add physics-like data to potentially trigger NIF
        system
    end

    start_time = System.monotonic_time(:microsecond)
    start_memory = :erlang.memory()

    try do
      # Add logging to track execution path
      IO.puts("    Starting execution...")

      # Run the simulation
      result = case mode do
        :interpreted ->
          IO.puts("    Using interpreted path (forced)")
          run_interpreted_simulation(modified_system, steps)
        :nif ->
          IO.puts("    Attempting NIF path...")
          run_nif_simulation(modified_system, steps)
      end

      end_time = System.monotonic_time(:microsecond)
      end_memory = :erlang.memory()

      execution_time = (end_time - start_time) / 1000  # Convert to ms
      memory_used = end_memory[:total] - start_memory[:total]

      IO.puts("    ✓ Completed in #{Float.round(execution_time, 2)}ms")
      IO.puts("    ✓ Memory used: #{div(memory_used, 1024)}KB")

      %{
        time: execution_time,
        memory: memory_used,
        status: "success",
        final_state: result,
        execution_path: determine_execution_path(result, mode)
      }

    rescue
      e ->
        IO.puts("    ✗ Failed: #{Exception.message(e)}")
        %{
          time: nil,
          memory: nil,
          status: "failed: #{Exception.message(e)}",
          error: e,
          execution_path: "error"
        }
    end
  end

  defp run_interpreted_simulation(system, steps) do
    # Force interpreted execution by calling runtime directly with rules that prevent NIF
    opts = [
      steps: steps,
      dt: 0.01,
      rules: [:integrate_no_gravity]  # Add rule to prevent NIF optimization
    ]

    # Use the actor model's simulation directly
    ActorModel.run_simulation(system, opts)
  end

  defp run_nif_simulation(system, steps) do
    # Try to use NIF path - this will likely fail for actor models
    # but we want to document this behavior
    opts = [
      steps: steps,
      dt: 0.01
    ]

    ActorModel.run_simulation(system, opts)
  end

  defp determine_execution_path(result, intended_mode) do
    # Analyze result to determine actual execution path
    cond do
      result == nil ->
        "unknown"
      is_map(result) and Map.has_key?(result, :actors) ->
        # Check if it looks like actor model result
        if Map.has_key?(result, :actors) do
          "actor_interpreted"
        else
          "unknown"
        end
      is_map(result) and Map.has_key?(result, :particles) ->
        "physics_nif"
      true ->
        "#{intended_mode}_unknown"
    end
  end

  defp generate_comparison_report(results) do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("COMPARISON REPORT")
    IO.puts(String.duplicate("=", 80))

    Enum.each(results, fn result ->
      IO.puts("\n#{result.name}:")
      IO.puts("  Actors: #{result.actor_count}, Steps: #{result.steps}")

      # Interpreted results
      if result.interpreted.time do
        IO.puts("  Interpreted: #{Float.round(result.interpreted.time, 2)}ms " <>
                "(#{div(result.interpreted.memory, 1024)}KB) - #{result.interpreted.status}")
        IO.puts("    Path: #{result.interpreted.execution_path}")
      else
        IO.puts("  Interpreted: Failed - #{result.interpreted.status}")
      end

      # NIF results
      if result.nif.time do
        IO.puts("  NIF: #{Float.round(result.nif.time, 2)}ms " <>
                "(#{div(result.nif.memory, 1024)}KB) - #{result.nif.status}")
        IO.puts("    Path: #{result.nif.execution_path}")

        # Performance comparison
        if result.interpreted.time && result.nif.time do
          speedup = result.interpreted.time / result.nif.time
          IO.puts("    Speedup: #{Float.round(speedup, 2)}x")
        end
      else
        IO.puts("  NIF: #{result.nif.status}")
      end
    end)

    # Summary
    successful_interpreted = Enum.count(results, fn r -> r.interpreted.time != nil end)
    successful_nif = Enum.count(results, fn r -> r.nif.time != nil end)

    IO.puts("\nSUMMARY:")
    IO.puts("  Successful interpreted runs: #{successful_interpreted}/#{length(results)}")
    IO.puts("  Successful NIF runs: #{successful_nif}/#{length(results)}")

    if successful_nif > 0 do
      avg_speedup = results
        |> Enum.filter(fn r -> r.interpreted.time != nil and r.nif.time != nil end)
        |> Enum.map(fn r -> r.interpreted.time / r.nif.time end)
        |> Enum.sum()
        |> Kernel./(successful_nif)

      IO.puts("  Average NIF speedup: #{Float.round(avg_speedup, 2)}x")
    end
  end

  defp run_detailed_benchmarks do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("DETAILED BENCHMARKS (Benchee)")
    IO.puts(String.duplicate("=", 80))

    # Create a test system for detailed benchmarking
    test_system = create_test_system(50)

    # Benchmark different operations
    Benchee.run(
      %{
        "Actor Creation (50 actors)" => fn ->
          create_test_system(50)
        end,

        "Message Processing (100 messages)" => fn ->
          system = create_test_system(10)
          messages = for i <- 1..100 do
            %{
              sender: 9999,
              receiver: rem(i, 10) + 1,
              content: "test #{i}",
              type: :increment,
              timestamp: :erlang.system_time(:millisecond) / 1000.0
            }
          end
          %{system | messages: messages}
          |> ActorModel.run_simulation(steps: 10, dt: 0.01)
        end,

        "Supervisor Operations" => fn ->
          system = create_test_system(20)
          # Simulate supervisor restarts
          updated_supervisors = Enum.map(system.supervisors, fn sup ->
            %{sup | restart_count: sup.restart_count + 1}
          end)
          %{system | supervisors: updated_supervisors}
          |> ActorModel.run_simulation(steps: 50, dt: 0.01)
        end,

        "Scheduler Load Balancing" => fn ->
          system = create_test_system(100)
          # Simulate scheduler operations
          updated_scheduler = %{system.scheduler |
            ready_queue: Enum.map(1..50, & &1),
            running_actors: Enum.map(51..100, & &1)
          }
          %{system | scheduler: updated_scheduler}
          |> ActorModel.run_simulation(steps: 100, dt: 0.01)
        end
      },
      memory_time: 2,
      time: 5,
      warmup: 1,
      print: [configuration: false],
      formatters: [
        {Benchee.Formatters.Console, extended_statistics: true},
        {Benchee.Formatters.HTML, file: "benchmarks/actor_model_comparison_detailed.html"}
      ]
    )
  end
end

# Run the comparison benchmark
ActorModelComparisonBench.run()
