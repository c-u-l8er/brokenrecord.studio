defmodule Examples.ExamplesTest do
  use ExUnit.Case

  test "examples modules exist" do
    # Test that example modules can be loaded
    assert Code.ensure_loaded?(Examples.ChemicalReactionNet)
    assert Code.ensure_loaded?(Examples.GravitySimulation)
    assert Code.ensure_loaded?(MyPhysics)
  end

  test "example modules have expected functions" do
    # Test that example modules export expected functions
    assert function_exported?(Examples.ChemicalReactionNet, :chemical_mixture, 4)
    assert function_exported?(Examples.GravitySimulation, :solar_system, 0)
    assert Code.ensure_loaded?(MyPhysics.CollisionWorld)
  end
end
