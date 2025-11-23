defmodule Examples.TestHelper do
  @moduledoc """
  Common test utilities for example tests.
  """

  import ExUnit.Assertions

  @doc """
  Asserts that two floating point values are approximately equal within tolerance.
  """
  def assert_approx_equal(actual, expected, tolerance \\ 0.001) when is_number(actual) and is_number(expected) do
    assert abs(actual - expected) <= tolerance,
           "Expected #{expected} ± #{tolerance}, got #{actual}"
  end

  @doc """
  Asserts that two 3D vectors are approximately equal within tolerance.
  """
  def assert_vectors_equal({x1, y1, z1}, {x2, y2, z2}, tolerance \\ 0.001) do
    assert_approx_equal(x1, x2, tolerance)
    assert_approx_equal(y1, y2, tolerance)
    assert_approx_equal(z1, z2, tolerance)
  end

  @doc """
  Asserts that a conservation law holds within tolerance.
  """
  def assert_conservation(initial_value, final_value, tolerance \\ 0.01, name \\ "quantity") do
    error = abs(final_value - initial_value) / abs(initial_value)
    assert error <= tolerance,
           "#{name} not conserved: initial=#{initial_value}, final=#{final_value}, error=#{error}"
  end

  @doc """
  Creates a mock state for testing with basic structure.
  """
  def mock_state(agents \\ []) do
    %{
      agents: agents,
      timestamp: :erlang.system_time(:millisecond) / 1000.0
    }
  end

  @doc """
  Creates a mock particle with basic properties.
  """
  def mock_particle(opts \\ []) do
    %{
      position: Keyword.get(opts, :position, {0.0, 0.0, 0.0}),
      velocity: Keyword.get(opts, :velocity, {0.0, 0.0, 0.0}),
      mass: Keyword.get(opts, :mass, 1.0),
      radius: Keyword.get(opts, :radius, 1.0),
      id: Keyword.get(opts, :id, "test_particle")
    }
  end

  @doc """
  Creates a mock body for gravity simulations.
  """
  def mock_body(opts \\ []) do
    %{
      position: Keyword.get(opts, :position, {0.0, 0.0, 0.0}),
      velocity: Keyword.get(opts, :velocity, {0.0, 0.0, 0.0}),
      mass: Keyword.get(opts, :mass, 1.0),
      radius: Keyword.get(opts, :radius, 1.0)
    }
  end

  @doc """
  Creates a mock molecule for chemical reaction simulations.
  """
  def mock_molecule(opts \\ []) do
    %{
      position: Keyword.get(opts, :position, {0.0, 0.0, 0.0}),
      velocity: Keyword.get(opts, :velocity, {0.0, 0.0, 0.0}),
      mass: Keyword.get(opts, :mass, 1.0),
      radius: Keyword.get(opts, :radius, 0.5),
      chemical_type: Keyword.get(opts, :chemical_type, :A),
      energy: Keyword.get(opts, :energy, 20.0)
    }
  end

  @doc """
  Creates a mock actor for actor model simulations.
  """
  def mock_actor(opts \\ []) do
    %{
      pid: Keyword.get(opts, :pid, 1),
      state: Keyword.get(opts, :state, %{}),
      mailbox: Keyword.get(opts, :mailbox, []),
      behavior: Keyword.get(opts, :behavior, :worker),
      supervisor: Keyword.get(opts, :supervisor, 0),
      status: Keyword.get(opts, :status, :running),
      processing_time: Keyword.get(opts, :processing_time, 0.0)
    }
  end

  @doc """
  Calculates distance between two 3D points.
  """
  def distance({x1, y1, z1}, {x2, y2, z2}) do
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    :math.sqrt(dx*dx + dy*dy + dz*dz)
  end

  @doc """
  Calculates dot product of two 3D vectors.
  """
  def dot({x1, y1, z1}, {x2, y2, z2}) do
    x1 * x2 + y1 * y2 + z1 * z2
  end

  @doc """
  Normalizes a 3D vector.
  """
  def normalize({x, y, z}) do
    length = :math.sqrt(x*x + y*y + z*z)
    if length > 0.0 do
      {x/length, y/length, z/length}
    else
      {0.0, 0.0, 0.0}
    end
  end

  @doc """
  Measures execution time of a function.
  """
  def measure_time(fun) when is_function(fun, 0) do
    {time, result} = :timer.tc(fun)
    {time / 1000.0, result}  # Convert to milliseconds
  end

  @doc """
  Asserts that a simulation completes within reasonable time.
  """
  def assert_performance(fun, max_time_ms \\ 1000) when is_function(fun, 0) do
    {time, _result} = measure_time(fun)
    assert time <= max_time_ms,
           "Simulation took too long: #{time}ms > #{max_time_ms}ms"
  end
end
