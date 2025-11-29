defmodule Examples.AIIChemicalReactionsTest do
  use ExUnit.Case
  alias Examples.AIIChemicalReactions
  import Examples.TestHelper

  setup_all do
    # Ensure DSL modules are loaded for testing
    Code.require_file("lib/aii/types.ex", ".")
    Code.require_file("lib/aii/dsl.ex", ".")
    Code.require_file("lib/examples/aii_chemical_reactions.ex", ".")
    :ok
  end

  describe "create_reaction_system/1" do
    test "creates system with default number of molecules" do
      system = AIIChemicalReactions.create_reaction_system()

      assert is_map(system)
      assert Map.has_key?(system, :molecules)
      assert Map.has_key?(system, :environment)
      assert Map.has_key?(system, :time)
      assert Map.has_key?(system, :step)

      assert length(system.molecules) == 20
      assert system.time == 0.0
      assert system.step == 0
    end

    test "creates system with custom number of molecules" do
      system = AIIChemicalReactions.create_reaction_system(10)

      assert length(system.molecules) == 10
    end

    test "molecules have correct structure" do
      system = AIIChemicalReactions.create_reaction_system(1)
      molecule = hd(system.molecules)

      # Check required fields
      required_fields = [:molecular_formula, :molecular_weight, :molecule_type,
                        :position, :velocity, :temperature, :concentration, :bonds,
                        :mass, :charge, :energy, :atoms, :information]

      Enum.each(required_fields, fn field ->
        assert Map.has_key?(molecule, field), "Molecule missing field: #{field}"
      end)

      # Check types
      assert is_binary(molecule.molecular_formula)
      assert is_float(molecule.molecular_weight)
      assert molecule.molecule_type in [:reactant, :product, :catalyst]
      assert is_tuple(molecule.position)
      assert is_tuple(molecule.velocity)
      assert is_float(molecule.temperature)
      assert is_float(molecule.concentration)
      assert is_list(molecule.bonds)
    end
  end

  describe "run_simulation/2" do
    test "runs simulation with default options" do
      initial_state = AIIChemicalReactions.create_reaction_system()

      result = AIIChemicalReactions.run_simulation(initial_state)

      assert is_map(result)
      assert result.steps == 1000
      assert result.dt == 0.001
      assert is_list(result.molecules)
    end

    test "runs simulation with custom options" do
      initial_state = AIIChemicalReactions.create_reaction_system(3)

      result = AIIChemicalReactions.run_simulation(initial_state, steps: 100, dt: 0.01)

      assert is_map(result)
      assert result.steps == 100
      assert result.dt == 0.01
    end
  end

  describe "reaction_stats/1" do
    test "calculates reaction statistics" do
      system = AIIChemicalReactions.create_reaction_system(5)

      stats = AIIChemicalReactions.reaction_stats(system)

      assert is_map(stats)
      assert Map.has_key?(stats, :total_molecules)
      assert Map.has_key?(stats, :total_mass)
      assert Map.has_key?(stats, :total_charge)
      assert Map.has_key?(stats, :total_energy)
      assert Map.has_key?(stats, :reaction_count)
      assert Map.has_key?(stats, :catalyst_efficiency)

      assert stats.total_molecules == 5
      assert is_float(stats.total_mass)
      assert is_float(stats.total_charge)
      assert is_float(stats.total_energy)
    end

    test "handles empty system" do
      empty_system = %{molecules: [], environment: %{}, time: 0.0, step: 0}

      stats = AIIChemicalReactions.reaction_stats(empty_system)

      assert stats.total_molecules == 0
      assert stats.total_mass == 0.0
      assert stats.total_charge == 0.0
      assert stats.total_energy == 0.0
    end
  end

  describe "helper functions" do
    test "molecular_weight calculates correct weight" do
      # H2O should have molecular weight ~18
      assert_in_delta AIIChemicalReactions.molecular_weight(%{H: 2, O: 1}), 18.0, 1.0

      # CO2 should have molecular weight ~44
      assert_in_delta AIIChemicalReactions.molecular_weight(%{C: 1, O: 2}), 44.0, 1.0
    end

    test "net_charge calculates correct charge" do
      # Neutral molecule
      assert AIIChemicalReactions.net_charge(%{"H+": 1, "OH-": 1}) == 0

      # Positive ion
      assert AIIChemicalReactions.net_charge(%{"H+": 2, "OH-": 1}) == 1

      # Negative ion
      assert AIIChemicalReactions.net_charge(%{"H+": 1, "OH-": 2}) == -1
    end

    test "generate_bonds creates proper bond structure" do
      formula = %{C: 1, H: 4}  # Methane
      bonds = AIIChemicalReactions.generate_bonds(formula)

      assert is_list(bonds)
      # Methane should have 4 C-H bonds
      assert length(bonds) == 4
    end
  end

  describe "DSL integration" do
    test "module has correct agents and interactions" do
      # Ensure the module is loaded and compiled
      assert function_exported?(AIIChemicalReactions, :__agents__, 0),
        "Module not properly compiled with DSL. Check that lib/examples/ are included in compilation."

      assert function_exported?(AIIChemicalReactions, :__interactions__, 0),
        "Module not properly compiled with DSL. Check that lib/examples/ are included in compilation."

      agents = AIIChemicalReactions.__agents__()
      interactions = AIIChemicalReactions.__interactions__()

      assert is_list(agents)
      assert is_list(interactions)

      # Should have at least Molecule agent
      assert length(agents) >= 1

      # Should have bonding, reaction, catalytic, and diffusion interactions
      assert length(interactions) >= 4
    end
  end

  test "conservation verification works" do
    # Test that the DSL conservation checker works
    interaction = %{body: {:chemical_bonding, [], []}, name: :chemical_bonding}
    agents = AIIChemicalReactions.__agents__()

    result = AII.verify_conservation(interaction, agents)

    # Should not crash and return some result
    assert result == :ok or match?({:needs_runtime_check, _, _}, result)
  end

  describe "performance" do
    test "simulation completes in reasonable time" do
      system = AIIChemicalReactions.create_reaction_system(5)

      {time_ms, result} = measure_time(fn ->
        AIIChemicalReactions.run_simulation(system, steps: 50)
      end)

      assert is_map(result)
      # Should complete in reasonable time for 5 molecules, 50 steps
      assert time_ms < 5000
    end

    test "statistics calculation is fast" do
      system = AIIChemicalReactions.create_reaction_system(10)

      {time_ms, _} = measure_time(fn ->
        AIIChemicalReactions.reaction_stats(system)
      end)

      # Should be very fast
      assert time_ms < 10
    end
  end

  describe "edge cases" do
    test "handles single molecule system" do
      system = AIIChemicalReactions.create_reaction_system(1)

      result = AIIChemicalReactions.run_simulation(system, steps: 10)
      assert is_map(result)

      stats = AIIChemicalReactions.reaction_stats(system)
      assert stats.total_molecules == 1
    end

    test "handles empty system" do
      system = %{molecules: [], environment: %{}, time: 0.0, step: 0}

      result = AIIChemicalReactions.run_simulation(system, steps: 5)
      expected = %{
        time: 0.005,
        step: 5,
        steps: 5,
        dt: 0.001,
        molecules: [],
        environment: %{},
        interactions_applied: []
      }
      assert result == expected

      stats = AIIChemicalReactions.reaction_stats(system)
      assert stats.total_molecules == 0
    end

    test "molecular_weight handles empty formula" do
      assert AIIChemicalReactions.molecular_weight(%{}) == 0.0
    end

    test "net_charge handles empty formula" do
      assert AIIChemicalReactions.net_charge(%{}) == 0
    end
  end
end
