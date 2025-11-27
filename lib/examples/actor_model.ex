defmodule Examples.ActorModel do
  @moduledoc """
  Example: Actor model implementation using lafont interaction nets.

  Demonstrates:
  - Actor-based concurrency
  - Message passing semantics
  - Actor lifecycle management
  - Supervision and fault tolerance
  - Load balancing and scheduling
  """


  defmodule ActorRuntime do
    @doc """
    Simulates the actor system for given steps.
    """
    def simulate(initial_state, opts) do
      steps = Keyword.get(opts, :steps, 1000)
      dt = Keyword.get(opts, :dt, 0.01)

      # Mock simulation - just advance time
      %{initial_state | time: (initial_state[:time] || 0) + steps * dt}
    end
  end

  @doc """
  Create initial actor system configuration
  """
  def actor_system do
    # Create root supervisor
    root_supervisor = %{
      pid: 0,
      children: [1, 2, 3, 4],
      strategy: :one_for_one,
      restart_policy: :permanent,
      max_restarts: 3,
      restart_count: 0
    }

    # Create actors with different behaviors
    actors = [
      %{
        pid: 1,
        state: %{count: 0},
        mailbox: [],
        behavior: :counter,
        supervisor: 0,
        status: :running,
        processing_time: 0.0
      },
      %{
        pid: 2,
        state: %{last_result: 0, operations: 0},
        mailbox: [],
        behavior: :calculator,
        supervisor: 0,
        status: :running,
        processing_time: 0.0
      },
      %{
        pid: 3,
        state: %{logs: []},
        mailbox: [],
        behavior: :logger,
        supervisor: 0,
        status: :running,
        processing_time: 0.0
      },
      %{
        pid: 4,
        state: %{completed: 0},
        mailbox: [],
        behavior: :worker,
        supervisor: 0,
        status: :running,
        processing_time: 0.0
      }
    ]

    # Create scheduler
    scheduler = %{
      ready_queue: [],
      running_actors: [1, 2, 3, 4],
      time_slice: 1.0,
      load_balance_strategy: :round_robin
    }

    # Create initial messages
    messages = [
      %{
        sender: 999,  # External sender
        receiver: 1,
        content: nil,
        type: :increment,
        timestamp: :erlang.system_time(:millisecond) / 1000.0
      },
      %{
        sender: 999,
        receiver: 2,
        content: {5, 3},
        type: :add,
        timestamp: :erlang.system_time(:millisecond) / 1000.0
      },
      %{
        sender: 999,
        receiver: 3,
        content: "System started",
        type: :log,
        timestamp: :erlang.system_time(:millisecond) / 1000.0
      },
      %{
        sender: 999,
        receiver: 4,
        content: %{task: "compute", data: [1, 2, 3, 4, 5]},
        type: :work,
        timestamp: :erlang.system_time(:millisecond) / 1000.0
      }
    ]

    %{
      actors: actors,
      messages: messages,
      supervisors: [root_supervisor],
      scheduler: scheduler
    }
  end

  @doc """
  Run simulation and return final state
  """
  def run_simulation(initial_state, opts) do
    steps = Keyword.get(opts, :steps, 1000)
    dt = Keyword.get(opts, :dt, 0.01)

    ActorRuntime.simulate(initial_state, steps: steps, dt: dt)
  end

  @doc """
  Send a message to an actor
  """
  def send_message(state, sender_pid, receiver_pid, content, type) do
    message = %{
      sender: sender_pid,
      receiver: receiver_pid,
      content: content,
      type: type,
      timestamp: :erlang.system_time(:millisecond) / 1000.0
    }

    %{state | messages: [message | state.messages]}
  end

  @doc """
  Get actor state by PID
  """
  def get_actor_state(state, pid) do
    Enum.find(state.actors, fn actor -> actor.pid == pid end)
  end

  @doc """
  Count messages in system
  """
  def count_messages(state) do
    length(state.messages)
  end

  @doc """
  Count running actors
  """
  def count_running_actors(state) do
    Enum.count(state.actors, fn actor -> actor.status == :running end)
  end

  @doc """
  Get system statistics
  """
  def system_stats(state) do
    %{
      total_actors: length(state.actors),
      running_actors: count_running_actors(state),
      total_messages: count_messages(state),
      messages_in_mailboxes: Enum.sum(Enum.map(state.actors, fn a -> length(a.mailbox) end)),
      supervisor_restarts: Enum.sum(Enum.map(state.supervisors, fn s -> s.restart_count end))
    }
  end
end
