defmodule AII.TypesTest do
  use ExUnit.Case
  alias AII.Types.{Conserved, Energy, Momentum, Vec3}

  test "Conserved.new creates tracked value" do
    c = Conserved.new(10.0, :test)
    assert c.value == 10.0
    assert c.source == :test
    assert c.tracked == true
  end

  test "Conserved.transfer moves value" do
    from = Conserved.new(10.0, :initial)
    to = Conserved.new(0.0, :target)

    {:ok, new_from, new_to} = Conserved.transfer(from, to, 3.0)
    assert new_from.value == 7.0
    assert new_to.value == 3.0
  end

  test "Conserved.transfer fails on insufficient value" do
    from = Conserved.new(2.0, :initial)
    to = Conserved.new(0.0, :target)

    assert {:error, :insufficient_value} = Conserved.transfer(from, to, 5.0)
  end

  test "Vec3.add" do
    v1 = {1.0, 2.0, 3.0}
    v2 = {4.0, 5.0, 6.0}
    assert Vec3.add(v1, v2) == {5.0, 7.0, 9.0}
  end

  test "Vec3.mul" do
    v = {2.0, 3.0, 4.0}
    assert Vec3.mul(v, 2.0) == {4.0, 6.0, 8.0}
  end

  test "Vec3.magnitude" do
    v = {3.0, 4.0, 0.0}
    assert Vec3.magnitude(v) == 5.0
  end

  test "Vec3.sub" do
    v1 = {5.0, 7.0, 9.0}
    v2 = {1.0, 2.0, 3.0}
    assert Vec3.sub(v1, v2) == {4.0, 5.0, 6.0}
  end

  test "Vec3.cross" do
    v1 = {1.0, 0.0, 0.0}
    v2 = {0.0, 1.0, 0.0}
    assert Vec3.cross(v1, v2) == {0.0, 0.0, 1.0}
  end

  test "Vec3.dot" do
    v1 = {1.0, 2.0, 3.0}
    v2 = {4.0, 5.0, 6.0}
    assert Vec3.dot(v1, v2) == 32.0
  end
end
