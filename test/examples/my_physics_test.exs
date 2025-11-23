defmodule Examples.MyPhysicsTest do
  use ExUnit.Case
  alias MyPhysics.CollisionWorld
  import Examples.TestHelper

  defp mock_wall(opts \\ []) do
    %{
      id: Keyword.get(opts, :id, "test_wall"),
      position: Keyword.get(opts, :position, {0.0, 0.0, 0.0}),
      normal: Keyword.get(opts, :normal, {0.0, 0.0, 1.0})
    }
  end

  defp total_kinetic_energy(%{particles: particles}) do
    Enum.reduce(particles, 0.0, fn
      %{velocity: {vx, vy, vz}, mass: m}, acc ->
        acc + 0.5 * m * (vx*vx + vy*vy + vz*vz)
    end)
  end

  defp total_momentum(%{particles: particles}) do
    Enum.reduce(particles, {0.0, 0.0, 0.0}, fn
      %{velocity: {vx, vy, vz}, mass: m}, {px, py, pz} ->
        {px + vx * m, py + vy * m, pz + vz * m}
    end)
  end

  describe "agent structure" do
    test "particles have required fields" do
      particle = mock_particle()
      assert is_binary(particle.id)
      assert is_tuple(particle.position) and tuple_size(particle.position) == 3
      assert is_tuple(particle.velocity) and tuple_size(particle.velocity) == 3
      assert is_number(particle.mass) and particle.mass > 0
      assert is_number(particle.radius) and particle.radius > 0
    end

    test "walls have required fields" do
      wall = mock_wall()
      assert is_binary(wall.id)
      assert is_tuple(wall.position) and tuple_size(wall.position) == 3
      assert is_tuple(wall.normal) and tuple_size(wall.normal) == 3
    end
  end

  describe "gravity integration" do
    test "gravity accelerates particles downward" do
      particle = mock_particle(position: {0.0, 0.0, 10.0}, velocity: {0.0, 0.0, 0.0})
      initial = %{particles: [particle], walls: []}
      final = CollisionWorld.simulate(initial, steps: 200, dt: 0.01)
      [f_particle] = final.particles
      {_, _, vz} = f_particle.velocity
      assert_approx_equal(vz, -1.962, 0.1)
      {_, _, z} = f_particle.position
      assert z < 9.0
    end

    test "positions update according to velocity" do
      particle = mock_particle(position: {0.0, 0.0, 0.0}, velocity: {1.0, 2.0, 0.0})
      initial = %{particles: [particle], walls: []}
      final = CollisionWorld.simulate(initial, steps: 10, dt: 0.1)
      [f_particle] = final.particles
      {x, y, _} = f_particle.position
      assert_approx_equal(x, 1.0, 0.1)
      assert_approx_equal(y, 2.0, 0.1)
    end
  end

  describe "particle-particle collisions" do
    test "head-on elastic collision swaps velocities (equal mass)" do
      p1 = mock_particle(id: "p1", position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0, radius: 0.4)
      p2 = mock_particle(id: "p2", position: {1.5, 0.0, 0.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0, radius: 0.4)
      initial = %{particles: [p1, p2], walls: []}
      final = CollisionWorld.simulate(initial, steps: 1000, dt: 0.001)
      p1_final = Enum.find(final.particles, &(&1.id == "p1"))
      p2_final = Enum.find(final.particles, &(&1.id == "p2"))
      assert_vectors_equal(p1_final.velocity, {-1.0, 0.0, 0.0}, 0.3)
      assert_vectors_equal(p2_final.velocity, {1.0, 0.0, 0.0}, 0.3)
    end

    test "collision conserves kinetic energy" do
      p1 = mock_particle(id: "p1", position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0, radius: 0.4)
      p2 = mock_particle(id: "p2", position: {1.5, 0.0, 0.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0, radius: 0.4)
      initial = %{particles: [p1, p2], walls: []}
      final = CollisionWorld.simulate(initial, steps: 1000, dt: 0.001)
      initial_ke = total_kinetic_energy(initial)
      final_ke = total_kinetic_energy(final)
      assert_conservation(initial_ke, final_ke, 0.05, "kinetic energy")
    end

    test "collision conserves momentum" do
      p1 = mock_particle(id: "p1", position: {0.0, 0.0, 0.0}, velocity: {1.0, 0.0, 0.0}, mass: 1.0, radius: 0.4)
      p2 = mock_particle(id: "p2", position: {1.5, 0.0, 0.0}, velocity: {-1.0, 0.0, 0.0}, mass: 1.0, radius: 0.4)
      initial = %{particles: [p1, p2], walls: []}
      final = CollisionWorld.simulate(initial, steps: 1000, dt: 0.001)
      initial_mom = total_momentum(initial)
      final_mom = total_momentum(final)
      assert_vectors_equal(initial_mom, final_mom, 0.1)
    end
  end

  describe "wall bounces" do
    test "particle bounces off floor" do
      particle = mock_particle(id: "p1", position: {0.0, 0.0, 1.5}, velocity: {0.0, 0.0, -1.0}, mass: 1.0, radius: 0.6)
      wall = mock_wall(id: "floor", position: {0.0, 0.0, 0.0}, normal: {0.0, 0.0, 1.0})
      initial = %{particles: [particle], walls: [wall]}
      final = CollisionWorld.simulate(initial, steps: 400, dt: 0.001)
      [p_final] = final.particles
      {_, _, vz} = p_final.velocity
      assert vz > 0.5
      {_, _, z} = p_final.position
      assert z > 0.5
    end
  end

  describe "performance" do
    test "100 particles simulate quickly" do
      particles = Enum.map(1..100, fn i ->
        mock_particle(id: "p#{i}", position: {i * 1.0 - 50.0, 0.0, 10.0})
      end)
      initial = %{particles: particles, walls: []}
      assert_performance(fn ->
        CollisionWorld.simulate(initial, steps: 100, dt: 0.01)
      end, 2000)
    end
  end

  describe "edge cases" do
    test "empty system simulates" do
      initial = %{particles: [], walls: []}
      final = CollisionWorld.simulate(initial, steps: 10, dt: 0.01)
      assert final.particles == []
      assert final.walls == []
    end

    test "single particle no walls" do
      particle = mock_particle()
      initial = %{particles: [particle], walls: []}
      final = CollisionWorld.simulate(initial, steps: 10, dt: 0.01)
      assert length(final.particles) == 1
    end
  end
end
