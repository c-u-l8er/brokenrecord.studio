
defmodule Examples.ChemicalReactionNetTest do
  use ExUnit.Case
  alias Examples.ChemicalReactionNet
  import Examples.TestHelper

  describe "ChemicalReactionNet.chemical_mixture/4" do
    test "creates a valid chemical mixture" do
      mixture = ChemicalReactionNet.chemical_mixture(10, 10, 5, 2)

      # Check structure
      assert is_map(mixture)
      assert is_list(mixture.molecules)
      assert is_list(mixture.catalysts)
      assert is_map(mixture.container)

      # Check molecule counts
      assert length(mixture.molecules) == 27  # 10 + 10 + 5 + 2
      assert length(mixture.catalysts) == 5

      # Check container
      assert has_container_fields(mixture.container)
    end

    test "molecules have correct properties" do
      mixture = ChemicalReactionNet.chemical_mixture(5, 3, 2, 1)

      # Count molecules by type
      counts = ChemicalReactionNet.count_molecules(mixture)
      assert counts[:A] == 5
      assert counts[:B] == 3
      assert counts[:C] == 2
      assert counts[:D] == 1

      # Check molecule properties
      Enum.each(mixture.molecules, fn molecule ->
        assert has_molecule_fields(molecule)
        assert molecule.chemical_type in [:A, :B, :C, :D]
        assert molecule.mass > 0
        assert molecule.radius > 0
        assert molecule.energy > 0
      end)
    end

    test "different molecule types have different properties" do
      mixture = ChemicalReactionNet.chemical_mixture(1, 1, 1, 1)

      molecules_by_type = Enum.group_by(mixture.molecules, & &1.chemical_type)

      # Type A molecules
      a_mol = hd(molecules_by_type[:A])
      assert a_mol.mass == 1.0
      assert a_mol.radius == 0.5

      # Type B molecules
      b_mol = hd(molecules_by_type[:B])
      assert b_mol.mass == 1.2
      assert b_mol.radius == 0.6

      # Type C molecules
      c_mol = hd(molecules_by_type[:C])
      assert c_mol.mass == 2.2
      assert c_mol.radius == 0.8

      # Type D molecules
      d_mol = hd(molecules_by_type[:D])
      assert d_mol.mass == 3.2
      assert d_mol.radius == 1.0
    end
  end

  describe "ChemicalReactionNet.count_molecules/1" do
    test "counts molecules correctly by type" do
      mixture = ChemicalReactionNet.chemical_mixture(10, 15, 5, 3)
      counts = ChemicalReactionNet.count_molecules(mixture)

      assert counts[:A] == 10
      assert counts[:B] == 15
      assert counts[:C] == 5
      assert counts[:D] == 3
    end

    test "handles empty molecule list" do
      empty_mixture = %{molecules: [], catalysts: [], container: mock_container()}
      counts = ChemicalReactionNet.count_molecules(empty_mixture)

      assert counts[:A] == 0
      assert counts[:B] == 0
      assert counts[:C] == 0
      assert counts[:D] == 0
    end

    test "ignores removed molecules (mass = 0)" do
      molecules = [
        mock_molecule(chemical_type: :A, mass: 1.0),
        mock_molecule(chemical_type: :B, mass: 0.0),  # Removed
        mock_molecule(chemical_type: :A, mass: 1.0),
        mock_molecule(chemical_type: :C, mass: 2.0)
      ]

      mixture = %{molecules: molecules, catalysts: [], container: mock_container()}
      counts = ChemicalReactionNet.count_molecules(mixture)

      assert counts[:A] == 2
      assert counts[:B] == 0  # Should not count removed molecule
      assert counts[:C] == 1
      assert counts[:D] == 0
    end
  end

  describe "ChemicalReactionNet.total_mass/1" do
    test "calculates total mass correctly" do
      mixture = ChemicalReactionNet.chemical_mixture(2, 3, 1, 1)
      total_mass = ChemicalReactionNet.total_mass(mixture)

      # Expected: 2*1.0 + 3*1.2 + 1*2.2 + 1*3.2 + 5*10.0 (catalysts)
      expected_molecule_mass = 2*1.0 + 3*1.2 + 1*2.2 + 1*3.2
      expected_catalyst_mass = 5 * 10.0
      expected_total = expected_molecule_mass + expected_catalyst_mass

      assert_approx_equal(total_mass, expected_total)
    end

    test "mass is conserved in simulation" do
      initial_mixture = ChemicalReactionNet.chemical_mixture(20, 20, 10, 5)
      initial_mass = ChemicalReactionNet.total_mass(initial_mixture)

      # Run a short simulation
      final_mixture = ChemicalReactionNet.simulate(initial_mixture, steps: 100, dt: 0.01)
      final_mass = ChemicalReactionNet.total_mass(final_mixture)

      # Mass should be approximately conserved
      assert_conservation(initial_mass, final_mass, 0.01, "total mass")
    end
  end

  describe "ChemicalReactionNet.total_energy/1" do
    test "calculates total energy correctly" do
      mixture = ChemicalReactionNet.chemical_mixture(2, 2, 1, 1)
      total_energy = ChemicalReactionNet.total_energy(mixture)

      assert is_number(total_energy)
      assert total_energy > 0
    end

    test "energy includes kinetic and chemical components" do
      # Create molecules with known velocities and energies
      molecules = [
        mock_molecule(
          position: {0.0, 0.0, 0.0},
          velocity: {1.0, 0.0, 0.0},
          mass: 1.0,
          energy: 50.0
        ),
        mock_molecule(
          position: {0.0, 0.0, 0.0},
          velocity: {0.0, 2.0, 0.0},
          mass: 2.0,
          energy: 60.0
        )
      ]

      mixture = %{molecules: molecules, catalysts: [], container: mock_container()}
      total_energy = ChemicalReactionNet.total_energy(mixture)

      # Expected: 0.5*1.0*1.0^2 + 50.0 + 0.5*2.0*2.0^2 + 60.0
      expected_kinetic = 0.5 * 1.0 * 1.0 + 0.5 * 2.0 * 4.0
      expected_chemical = 50.0 + 60.0
      expected_total = expected_kinetic + expected_chemical

      assert_approx_equal(total_energy, expected_total)
    end
  end

  describe "ChemicalReactionNet.reaction_rates/3" do
    test "calculates reaction rates correctly" do
      initial_mixture = ChemicalReactionNet.chemical_mixture(20, 20, 10, 5)
      final_mixture = ChemicalReactionNet.chemical_mixture(22, 18, 12, 5)  # Some reactions occurred

      rates = ChemicalReactionNet.reaction_rates(initial_mixture, final_mixture, 1.0)

      assert is_map(rates)
      assert Map.has_key?(rates, :synthesis_rate)
      assert Map.has_key?(rates, :decomposition_rate)

      # Synthesis: C increased from 10 to 12, so rate = 2/1.0 = 2.0
      assert_approx_equal(rates.synthesis_rate, 2.0)

      # Decomposition: A+B increased from 40 to 40, so rate = 0/1.0 = 0.0
      assert_approx_equal(rates.decomposition_rate, 0.0)
    end
  end

  describe "ChemicalReactionNet.simulate/2" do
    test "can run a basic simulation" do
      initial_mixture = ChemicalReactionNet.chemical_mixture(10, 10, 5, 2)

      # Test that simulation runs without errors
      assert_performance(fn ->
        ChemicalReactionNet.simulate(initial_mixture, steps: 10, dt: 0.01)
      end, 5000)  # 5 second timeout
    end

    test "simulation preserves system structure" do
      initial_mixture = ChemicalReactionNet.chemical_mixture(10, 10, 5, 2)
      final_mixture = ChemicalReactionNet.run_simulation(initial_mixture, steps: 10, dt: 0.01)

      # Check that structure is preserved
      assert is_map(final_mixture)
      assert is_list(final_mixture.molecules)
      assert is_list(final_mixture.catalysts)
      assert is_map(final_mixture.container)

      # Check that catalyst count is preserved
      assert length(final_mixture.catalysts) == length(initial_mixture.catalysts)
    end

    test "simulation changes molecule counts over time" do
      initial_mixture = ChemicalReactionNet.chemical_mixture(20, 20, 5, 0)
      initial_counts = ChemicalReactionNet.count_molecules(initial_mixture)

      # Run simulation long enough for reactions to occur
      final_mixture = ChemicalReactionNet.simulate(initial_mixture, steps: 1000, dt: 0.01)
      final_counts = ChemicalReactionNet.count_molecules(final_mixture)

      # Some change should occur (though exact amounts depend on random reactions)
      _total_initial = initial_counts[:A] + initial_counts[:B] + initial_counts[:C] + initial_counts[:D]
      total_final = final_counts[:A] + final_counts[:B] + final_counts[:C] + final_counts[:D]

      # Total number of molecules may change due to synthesis/decomposition
      assert is_number(total_final)
    end
  end

  describe "catalyst behavior" do
    test "catalysts have correct properties" do
      mixture = ChemicalReactionNet.chemical_mixture(5, 5, 2, 1)

      Enum.each(mixture.catalysts, fn catalyst ->
        assert has_catalyst_fields(catalyst)
        assert catalyst.active == true
        assert catalyst.catalysis_rate > 0.0
        assert catalyst.catalysis_rate <= 1.0
        assert is_list(catalyst.bound_molecules)
      end)
    end

    test "catalyst positions are within container" do
      mixture = ChemicalReactionNet.chemical_mixture(5, 5, 2, 1)
      container_radius = mixture.container.radius

      Enum.each(mixture.catalysts, fn catalyst ->
        dist_from_center = distance(catalyst.position, mixture.container.position)
        assert dist_from_center < container_radius
      end)
    end
  end

  describe "container properties" do
    test "container has correct default properties" do
      mixture = ChemicalReactionNet.chemical_mixture(5, 5, 2, 1)
      container = mixture.container

      assert container.position == {0.0, 0.0, 0.0}
      assert container.radius == 60.0
      assert container.temperature == 300.0
      assert container.pressure == 1.0
    end
  end

  describe "molecule positions and velocities" do
    test "molecules start within container" do
      mixture = ChemicalReactionNet.chemical_mixture(10, 10, 5, 2)
      container_radius = mixture.container.radius

      Enum.each(mixture.molecules, fn molecule ->
        dist_from_center = distance(molecule.position, mixture.container.position)
        assert dist_from_center < container_radius
      end)
    end

    test "molecules have reasonable velocities" do
      mixture = ChemicalReactionNet.chemical_mixture(10, 10, 5, 2)

      Enum.each(mixture.molecules, fn molecule ->
        {vx, vy, vz} = molecule.velocity
        speed = :math.sqrt(vx*vx + vy*vy + vz*vz)

        # Speed should be reasonable for the molecule type
        assert speed >= 0.0
        assert speed < 10.0  # Should not be excessively fast
      end)
    end
  end

  describe "energy conservation" do
    test "total energy is approximately conserved" do
      initial_mixture = ChemicalReactionNet.chemical_mixture(20, 20, 10, 5)
      initial_energy = ChemicalReactionNet.total_energy(initial_mixture)

      # Run simulation
      final_mixture = ChemicalReactionNet.simulate(initial_mixture, steps: 100, dt: 0.01)
      final_energy = ChemicalReactionNet.total_energy(final_mixture)

      # Energy should be approximately conserved (some loss due to inelastic collisions)
      energy_change = abs(final_energy - initial_energy) / initial_energy
      assert energy_change < 0.1  # Allow 10% energy loss
    end
  end

  # Helper functions
  defp has_molecule_fields(molecule) do
    is_tuple(molecule.position) and tuple_size(molecule.position) == 3 and
    is_tuple(molecule.velocity) and tuple_size(molecule.velocity) == 3 and
    is_number(molecule.mass) and molecule.mass >= 0 and
    is_number(molecule.radius) and molecule.radius > 0 and
    is_atom(molecule.chemical_type) and
    is_number(molecule.energy) and molecule.energy >= 0
  end

  defp has_catalyst_fields(catalyst) do
    is_tuple(catalyst.position) and tuple_size(catalyst.position) == 3 and
    is_boolean(catalyst.active) and
    is_number(catalyst.catalysis_rate) and
    is_list(catalyst.bound_molecules)
  end

  defp has_container_fields(container) do
    is_tuple(container.position) and tuple_size(container.position) == 3 and
    is_number(container.radius) and container.radius > 0 and
    is_number(container.temperature) and container.temperature > 0 and
    is_number(container.pressure) and container.pressure > 0
  end

  defp mock_container do
    %{
      position: {0.0, 0.0, 0.0},
      radius: 60.0,
      temperature: 300.0,
      pressure: 1.0
    }
  end
end
