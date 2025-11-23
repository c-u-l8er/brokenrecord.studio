defmodule Examples.ActorModelTest do
  use ExUnit.Case
  alias Examples.ActorModel
  import Examples.TestHelper

  describe "ActorModel.actor_system/0" do
    test "creates a valid actor system configuration" do
      system = ActorModel.actor_system()

      # Check system structure
      assert is_map(system)
      assert is_list(system.actors)
      assert is_list(system.messages)
      assert is_list(system.supervisors)
      assert is_map(system.scheduler)

      # Check actors
      assert length(system.actors) == 4
      assert Enum.all?(system.actors, &has_actor_fields/1)

      # Check messages
      assert length(system.messages) == 4
      assert Enum.all?(system.messages, &has_message_fields/1)

      # Check supervisors
      assert length(system.supervisors) == 1
      supervisor = hd(system.supervisors)
      assert has_supervisor_fields(supervisor)

      # Check scheduler
      assert has_scheduler_fields(system.scheduler)
    end

    test "actors have different behaviors" do
      system = ActorModel.actor_system()
      behaviors = Enum.map(system.actors, & &1.behavior)

      assert :counter in behaviors
      assert :calculator in behaviors
      assert :logger in behaviors
      assert :worker in behaviors
    end

    test "all actors are initially running" do
      system = ActorModel.actor_system()

      assert Enum.all?(system.actors, fn actor ->
        actor.status == :running
      end)
    end
  end

  describe "ActorModel.send_message/5" do
    test "adds a new message to the system" do
      initial_state = ActorModel.actor_system()
      initial_count = length(initial_state.messages)

      new_state = ActorModel.send_message(
        initial_state,
        999,  # sender_pid
        1,    # receiver_pid
        "test message",
        :test
      )

      assert length(new_state.messages) == initial_count + 1

      new_message = hd(new_state.messages)
      assert new_message.sender == 999
      assert new_message.receiver == 1
      assert new_message.content == "test message"
      assert new_message.type == :test
      assert is_number(new_message.timestamp)
    end
  end

  describe "ActorModel.get_actor_state/2" do
    test "returns correct actor by PID" do
      system = ActorModel.actor_system()

      actor = ActorModel.get_actor_state(system, 1)
      assert actor.pid == 1
      assert actor.behavior == :counter

      actor = ActorModel.get_actor_state(system, 2)
      assert actor.pid == 2
      assert actor.behavior == :calculator
    end

    test "returns nil for non-existent PID" do
      system = ActorModel.actor_system()

      actor = ActorModel.get_actor_state(system, 999)
      assert actor == nil
    end
  end

  describe "ActorModel.count_messages/1" do
    test "counts messages correctly" do
      system = ActorModel.actor_system()
      initial_count = ActorModel.count_messages(system)

      # Add some messages
      new_state = ActorModel.send_message(system, 999, 1, "test1", :test)
      new_state = ActorModel.send_message(new_state, 999, 2, "test2", :test)

      assert ActorModel.count_messages(new_state) == initial_count + 2
    end
  end

  describe "ActorModel.count_running_actors/1" do
    test "counts running actors correctly" do
      system = ActorModel.actor_system()

      # All actors should be running initially
      assert ActorModel.count_running_actors(system) == 4

      # Simulate a terminated actor
      modified_actors = List.replace_at(system.actors, 0,
        put_in(hd(system.actors).status, :terminated))
      modified_system = %{system | actors: modified_actors}

      assert ActorModel.count_running_actors(modified_system) == 3
    end
  end

  describe "ActorModel.system_stats/1" do
    test "returns comprehensive system statistics" do
      system = ActorModel.actor_system()
      stats = ActorModel.system_stats(system)

      assert is_map(stats)
      assert Map.has_key?(stats, :total_actors)
      assert Map.has_key?(stats, :running_actors)
      assert Map.has_key?(stats, :total_messages)
      assert Map.has_key?(stats, :messages_in_mailboxes)
      assert Map.has_key?(stats, :supervisor_restarts)

      assert stats.total_actors == 4
      assert stats.running_actors == 4
      assert stats.total_messages == 4
      assert stats.messages_in_mailboxes == 0
      assert stats.supervisor_restarts == 0
    end
  end

  describe "ActorModel.simulate/2" do
    test "can run a basic simulation" do
      initial_state = ActorModel.actor_system()

      # Test that simulation runs without errors
      assert_performance(fn ->
        ActorModel.simulate(initial_state, steps: 10, dt: 0.01)
      end, 5000)  # 5 second timeout
    end

    test "simulation preserves system structure" do
      initial_state = ActorModel.actor_system()
      final_state = ActorModel.simulate(initial_state, steps: 10, dt: 0.01)

      # Check that structure is preserved
      assert is_map(final_state)
      assert is_list(final_state.actors)
      assert is_list(final_state.messages)
      assert is_list(final_state.supervisors)
      assert is_map(final_state.scheduler)

      # Check that actor count is preserved
      assert length(final_state.actors) == length(initial_state.actors)
    end
  end

  describe "actor behaviors" do
    test "counter actor behavior" do
      system = ActorModel.actor_system()
      counter_actor = Enum.find(system.actors, &(&1.behavior == :counter))

      assert counter_actor.state == %{count: 0}
      assert counter_actor.mailbox == []
    end

    test "calculator actor behavior" do
      system = ActorModel.actor_system()
      calculator_actor = Enum.find(system.actors, &(&1.behavior == :calculator))

      assert calculator_actor.state == %{last_result: 0, operations: 0}
      assert calculator_actor.mailbox == []
    end

    test "logger actor behavior" do
      system = ActorModel.actor_system()
      logger_actor = Enum.find(system.actors, &(&1.behavior == :logger))

      assert logger_actor.state == %{logs: []}
      assert logger_actor.mailbox == []
    end

    test "worker actor behavior" do
      system = ActorModel.actor_system()
      worker_actor = Enum.find(system.actors, &(&1.behavior == :worker))

      assert worker_actor.state == %{completed: 0}
      assert worker_actor.mailbox == []
    end
  end

  describe "message processing" do
    test "messages have correct structure" do
      system = ActorModel.actor_system()

      Enum.each(system.messages, fn message ->
        assert has_message_fields(message)
        assert is_integer(message.sender)
        assert is_integer(message.receiver)
        assert message.receiver in 1..4  # Should target one of our actors
        assert is_atom(message.type)
        assert is_number(message.timestamp)
      end)
    end

    test "initial messages target correct actors" do
      system = ActorModel.actor_system()

      # Check that messages are targeted to the right actors based on type
      increment_msg = Enum.find(system.messages, &(&1.type == :increment))
      assert increment_msg.receiver == 1  # Counter actor

      add_msg = Enum.find(system.messages, &(&1.type == :add))
      assert add_msg.receiver == 2  # Calculator actor

      log_msg = Enum.find(system.messages, &(&1.type == :log))
      assert log_msg.receiver == 3  # Logger actor

      work_msg = Enum.find(system.messages, &(&1.type == :work))
      assert work_msg.receiver == 4  # Worker actor
    end
  end

  # Helper functions
  defp has_actor_fields(actor) do
    is_integer(actor.pid) and
    is_map(actor.state) and
    is_list(actor.mailbox) and
    is_atom(actor.behavior) and
    is_integer(actor.supervisor) and
    is_atom(actor.status) and
    is_number(actor.processing_time)
  end

  defp has_message_fields(message) do
    is_integer(message.sender) and
    is_integer(message.receiver) and
    true and
    is_atom(message.type) and
    is_number(message.timestamp)
  end

  defp has_supervisor_fields(supervisor) do
    is_integer(supervisor.pid) and
    is_list(supervisor.children) and
    is_atom(supervisor.strategy) and
    is_atom(supervisor.restart_policy) and
    is_integer(supervisor.max_restarts) and
    is_integer(supervisor.restart_count)
  end

  defp has_scheduler_fields(scheduler) do
    is_list(scheduler.ready_queue) and
    is_list(scheduler.running_actors) and
    is_number(scheduler.time_slice) and
    is_atom(scheduler.load_balance_strategy)
  end
end
