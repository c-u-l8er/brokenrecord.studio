defmodule Examples.AIIChemicalReactions do
  @moduledoc """
  Example: AII-based chemical reaction simulation with conservation types.

  Demonstrates:
  - Chemical reaction networks with conservation
  - Molecular interactions and bonding
  - Conservation of mass, charge, and energy
  - Reaction kinetics and thermodynamics
  - Catalyst and enzyme interactions
  """

  use AII.DSL

  # Declare conserved quantities for chemical systems
  conserved_quantity :mass, type: :scalar, law: :sum
  conserved_quantity :charge, type: :scalar, law: :sum
  conserved_quantity :energy, type: :scalar, law: :sum
  conserved_quantity :atoms, type: :vector, law: :sum
  conserved_quantity :information, type: :scalar, law: :sum

  defagent Molecule do
    property :molecular_formula, String, invariant: true
    property :molecular_weight, Float, invariant: true
    property :molecule_type, Atom, invariant: true  # :reactant, :product, :catalyst

    state :position, AII.Types.Vec3
    state :velocity, AII.Types.Vec3
    state :temperature, Float
    state :concentration, Float
    state :bonds, List  # List of bonded atoms

    # Conserved quantities
    state :mass, AII.Types.Conserved
    state :charge, AII.Types.Conserved
    state :energy, AII.Types.Conserved
    state :atoms, AII.Types.Conserved
    state :information, AII.Types.Conserved

    # Derived quantities
    derives :kinetic_energy, AII.Types.Energy do
      # Translational kinetic energy
      0.5 * mass.value * AII.Types.Vec3.magnitude(velocity) ** 2
    end

    derives :thermal_energy, AII.Types.Energy do
      # Thermal energy: kT where k is Boltzmann constant
      1.38e-23 * temperature * 6.022e23  # k_B * N_A
    end

    derives :binding_energy, Energy do
      # Energy stored in chemical bonds
      Enum.reduce(bonds, 0.0, fn bond, acc ->
        acc + bond_energy(bond.type, bond.strength)
      end)
    end

    derives :total_energy, Energy do
      kinetic_energy + thermal_energy + binding_energy
    end

    derives :atom_count, Atoms do
      # Count atoms by element
      count_atoms_by_element(molecular_formula)
    end

    derives :entropy, Information do
      # Information entropy based on molecular complexity
      complexity = length(bonds) * :math.log(length(bonds) + 1)
      temperature * :math.log(temperature + 1) * complexity
    end

    conserves :mass, :charge, :energy, :atoms, :information
  end

  defagent Atom do
    property :element, Atom, invariant: true  # :H, :O, :C, :N, etc.
    property :atomic_number, Integer, invariant: true
    property :atomic_mass, Float, invariant: true
    property :valence_electrons, Integer, invariant: true

    state :position, Vec3
    state :velocity, Vec3
    state :electron_density, Float
    state :bonded_to, List  # References to bonded molecules

    # Conserved quantities
    state :mass, AII.Types.Conserved
    state :charge, AII.Types.Conserved
    state :energy, AII.Types.Conserved
    state :information, AII.Types.Conserved

    derives :kinetic_energy, Energy do
      0.5 * atomic_mass * Vec3.magnitude(velocity) ** 2
    end

    derives :electronegativity, Float do
      # Simplified electronegativity based on electron density
      electron_density * valence_electrons / atomic_number
    end

    conserves :mass, :charge, :energy, :information
  end

  defagent Catalyst do
    property :catalyst_type, Atom, invariant: true  # :enzyme, :metal, :acid
    property :active_sites, Integer, invariant: true
    property :activation_energy, Float, invariant: true

    state :temperature, Float
    state :bound_molecules, List
    state :efficiency, Float

    # Conserved quantities (catalyst is not consumed)
    state :energy, AII.Types.Conserved
    state :information, AII.Types.Conserved

    derives :turnover_frequency, Float do
      # How many reactions per unit time
      efficiency * active_sites * temperature / 300.0  # Normalized to room temp
    end

    conserves :energy, :information
  end

  defagent ReactionEnvironment do
    property :volume, Float, invariant: true
    property :pressure, Float, invariant: true
    property :temperature, Float, invariant: true

    state :ph, Float  # pH of solution
    state :ionic_strength, Float
    state :solvent_concentration, Float

    # Conserved quantities for the environment
    state :total_energy, AII.Types.Conserved
    state :total_mass, AII.Types.Conserved
    state :total_charge, AII.Types.Conserved

    derives :gibbs_free_energy, Energy do
      # ΔG = ΔH - TΔS (simplified)
      total_energy.value - temperature * 0.001  # Simplified entropy term
    end

    conserves :energy, :mass, :charge
  end

  # Chemical bonding interaction - uses RT Cores for spatial proximity
  definteraction :chemical_bonding, accelerator: :rt_cores do
    let {atom1, atom2, environment} do
      # RT Cores accelerate spatial queries for nearby atoms
      distance = AII.Types.Vec3.magnitude(AII.Types.Vec3.sub(atom1.position, atom2.position))
      bonding_distance = calculate_bonding_distance(atom1, atom2)

      if distance < bonding_distance and can_bond?(atom1, atom2) do
        # Form chemical bond
        bond_type = determine_bond_type(atom1, atom2)
        bond_strength = calculate_bond_strength(bond_type, environment)

        # Update atom states
        atom1.bonded_to = [atom2 | atom1.bonded_to]
        atom2.bonded_to = [atom1 | atom2.bonded_to]

        # Conservation of mass and charge
        # Atoms combine but total mass/charge conserved

        # Energy released in bond formation (exothermic)
        bond_energy = bond_energy(bond_type, bond_strength)
        energy_transfer = AII.Types.Conserved.transfer(
          environment.total_energy,
          atom1.energy,
          bond_energy * 0.5  # Half to each atom
        )

        case energy_transfer do
          {:ok, env_energy, atom1_energy} ->
            environment.total_energy = env_energy
            atom1.energy = atom1_energy

            # Transfer remaining to atom2
            energy_transfer2 = AII.Types.Conserved.transfer(
              atom1_energy,
              atom2.energy,
              bond_energy * 0.5
            )

            case energy_transfer2 do
              {:ok, atom1_final, atom2_energy} ->
                atom1.energy = atom1_final
                atom2.energy = atom2_energy
              {:error, _} ->
                # Conservation violation
                :error
            end
          {:error, _} ->
            # Conservation violation
            :error
        end

        # Information increases (more ordered state)
        info_gain = :math.log(bond_strength + 1)
        info_transfer1 = AII.Types.Conserved.transfer(
          environment.total_energy,  # Using environment as info source
          atom1.information,
          info_gain * 0.5
        )

        info_transfer2 = AII.Types.Conserved.transfer(
          atom1.information,
          atom2.information,
          info_gain * 0.5
        )

        case {info_transfer1, info_transfer2} do
          {{:ok, _, atom1_info}, {:ok, atom1_final, atom2_info}} ->
            atom1.information = atom1_info
            atom2.information = atom2_info
          _ ->
            # Conservation violation
            :error
        end
      end
    end
  end

  # Chemical reaction - uses Tensor Cores for reaction matrix operations
  definteraction :chemical_reaction, accelerator: :tensor_cores do
    let {reactants, products, catalyst, environment} do
      # Tensor Cores accelerate reaction rate calculations
      # Check if reaction conditions are met

      # Calculate reaction Gibbs free energy
      reactants_energy = Enum.sum(Enum.map(reactants, & &1.total_energy.value))
      products_energy = Enum.sum(Enum.map(products, & &1.total_energy.value))

      delta_g = products_energy - reactants_energy

      # Reaction proceeds if ΔG < 0 and catalyst present
      if delta_g < 0 and catalyst.efficiency > 0.0 do
        # Verify atom conservation
        reactant_atoms = sum_atom_vectors(reactants)
        product_atoms = sum_atom_vectors(products)

        if atoms_equal?(reactant_atoms, product_atoms) do
          # Reaction can proceed - conserve atoms

          # Update molecule states
          Enum.each(reactants, fn reactant ->
            reactant.molecule_type = :consumed
            reactant.concentration = reactant.concentration * 0.9  # Decrease as consumed
          end)

          Enum.each(products, fn product ->
            product.molecule_type = :product
            product.concentration = product.concentration * 1.1  # Increase as produced
          end)

          # Energy transfer
          energy_released = -delta_g  # Negative ΔG means energy released

          # Distribute energy to products and environment
          energy_distribution = distribute_reaction_energy(
            energy_released,
            products,
            environment
          )

          case energy_distribution do
            {:ok, updated_products, updated_env} ->
              # Update states
              Enum.each(updated_products, fn p ->
                p.energy = p.energy
                p.temperature = p.temperature + energy_released / (length(updated_products) * 100)  # Heat products
              end)

              environment.total_energy = updated_env

              # Catalyst gains information (learned from reaction)
              catalyst_info_gain = :math.log(abs(delta_g) + 1) * catalyst.efficiency
              info_transfer = AII.Types.Conserved.transfer(
                environment.total_energy,
                catalyst.information,
                catalyst_info_gain
              )

              case info_transfer do
                {:ok, env_info, catalyst_info} ->
                  environment.total_energy = env_info
                  catalyst.information = catalyst_info
                {:error, _} ->
                  # Conservation violation
                  :error
                  :error
              end
            {:error, _} ->
              # Conservation violation
              :error
          end
        else
          # Atom conservation violation - reaction cannot proceed
          record_conservation_violation(
            :atom_conservation,
            reactant_atoms,
            product_atoms,
            environment
          )
        end
      end
    end
  end

  # Catalytic reaction - uses NPU for learned reaction pathways
  definteraction :catalytic_reaction, accelerator: :npu do
    let {substrate, catalyst, environment} do
      # NPU accelerates reaction pathway prediction

      # Check if substrate can bind to catalyst
      if can_bind_to_catalyst?(substrate, catalyst) do
        # NPU predicts optimal reaction pathway
        pathway_prediction = neural_network_predict(
          catalyst,
          substrate,
          environment
        )

        case pathway_prediction do
          {:ok, pathway, confidence} when confidence > 0.8 ->
            # High confidence pathway - proceed with reaction

            # Calculate activation energy reduction
            activation_reduction = catalyst.activation_energy * catalyst.efficiency

            # Energy transfer from catalyst to substrate
            energy_transfer = AII.Types.Conserved.transfer(
              catalyst.energy,
              substrate.energy,
              activation_reduction
            )

            case energy_transfer do
              {:ok, catalyst_energy, substrate_energy} ->
                catalyst.energy = catalyst_energy
                substrate.energy = substrate_energy

                # Transform substrate to product
                product = transform_substrate(substrate, pathway)

                # Update catalyst state
                catalyst.bound_molecules = [product | catalyst.bound_molecules]
                catalyst.efficiency = catalyst.efficiency * 0.999  # Slight degradation

                # Information transfer - catalyst learns
                learning_gain = confidence * :math.log(confidence + 1)
                info_transfer = AII.Types.Conserved.transfer(
                  substrate.information,
                  catalyst.information,
                  learning_gain
                )

                case info_transfer do
                  {:ok, substrate_info, catalyst_info} ->
                    substrate.information = substrate_info
                    catalyst.information = catalyst_info
                  {:error, _} ->
                    # Conservation violation
                    :error
                end

                product
              {:error, _} ->
                # Conservation violation
                substrate
            end
          {:ok, _, _} ->
            # Low confidence - reaction may not proceed
            substrate
          {:error, _} ->
            # Prediction failed
            substrate
        end
      else
        substrate
      end
    end
  end

  # Diffusion and mixing - uses CPU for simple physics
  definteraction :molecular_diffusion, accelerator: :cpu do
    let {molecules, environment} do
      # Simple diffusion based on concentration gradients

      Enum.each(molecules, fn molecule ->
        # Calculate diffusion rate (Fick's law)
        diffusion_coefficient = calculate_diffusion_coefficient(molecule, environment)

        # Random walk component
        random_force = {
          (:rand.uniform() - 0.5) * diffusion_coefficient,
          (:rand.uniform() - 0.5) * diffusion_coefficient,
          (:rand.uniform() - 0.5) * diffusion_coefficient
        }

        # Update velocity
        molecule.velocity = Vec3.add(
          molecule.velocity,
          Vec3.mul(random_force, 0.016)  # dt = 0.016
        )

        # Update position
        molecule.position = Vec3.add(
          molecule.position,
          Vec3.mul(molecule.velocity, 0.016)
        )

        # Energy dissipation to environment
        energy_loss = 0.001 * Vec3.magnitude(molecule.velocity) ** 2
        energy_transfer = AII.Types.Conserved.transfer(
          molecule.energy,
          environment.total_energy,
          energy_loss
        )

        case energy_transfer do
          {:ok, molecule_energy, env_energy} ->
            molecule.energy = molecule_energy
            environment.total_energy = env_energy
          {:error, _} ->
            # Conservation violation
            :error
        end
      end)
    end
  end

  # Helper functions
  defp calculate_bonding_distance(atom1, atom2) do
    # Bonding distance based on atomic radii
    base_distance = atom1.atomic_mass * 0.1 + atom2.atomic_mass * 0.1

    # Adjust for electronegativity difference
    electronegativity_diff = abs(
      atom1.electronegativity - atom2.electronegativity
    )

    base_distance * (1.0 + electronegativity_diff * 0.1)
  end

  defp can_bond?(atom1, atom2) do
    # Check valence electron availability
    atom1.valence_electrons > length(atom1.bonded_to) and
    atom2.valence_electrons > length(atom2.bonded_to) and
    # Different elements can bond (simplified)
    atom1.element != atom2.element
  end

  defp determine_bond_type(atom1, atom2) do
    # Simplified bond type determination
    case {atom1.element, atom2.element} do
      {:H, :O} -> :covalent
      {:C, :H} -> :covalent
      {:N, :H} -> :covalent
      {:O, :H} -> :covalent
      {:C, :O} -> :double_bond
      {:C, :N} -> :triple_bond
      _ -> :ionic
    end
  end

  defp calculate_bond_strength(bond_type, environment) do
    base_strength = case bond_type do
      :covalent -> 350.0  # kJ/mol
      :double_bond -> 600.0
      :triple_bond -> 800.0
      :ionic -> 400.0
    end

    # Adjust for temperature
    temperature_factor = :math.exp(-environment.temperature / 1000.0)
    base_strength * temperature_factor
  end

  defp bond_energy(bond_type, bond_strength) do
    # Energy released when bond forms
    case bond_type do
      :covalent -> -bond_strength * 0.8  # 80% released
      :double_bond -> -bond_strength * 0.7
      :triple_bond -> -bond_strength * 0.6
      :ionic -> -bond_strength * 0.9
    end
  end

  defp count_atoms_by_element(formula) do
    # Simple parser for chemical formulas like H2O, CO2, etc.
    Regex.scan(~r/([A-Z][a-z]*)(\d*)/, formula)
    |> Enum.map(fn [_full, element, count_str] ->
      atom_count = if count_str == "", do: 1, else: String.to_integer(count_str)
      {element, atom_count}
    end)
    |> Enum.into(%{})
  end

  defp sum_atom_vectors(molecules) do
    Enum.reduce(molecules, %{}, fn molecule, acc ->
      atom_counts = molecule.atom_count

      Enum.reduce(atom_counts, acc, fn {element, count}, inner_acc ->
        current = Map.get(inner_acc, element, 0)
        Map.put(inner_acc, element, current + count)
      end)
    end)
  end

  defp atoms_equal?(atoms1, atoms2) do
    # Check if atom vectors are equal
    elements1 = Map.keys(atoms1) |> Enum.sort()
    elements2 = Map.keys(atoms2) |> Enum.sort()

    elements1 == elements2 and
    Enum.all?(elements1, fn element ->
      Map.get(atoms1, element) == Map.get(atoms2, element)
    end)
  end

  defp distribute_reaction_energy(energy_released, products, environment) do
    # Distribute released energy among products and environment
    num_products = length(products)
    product_share = energy_released * 0.7  # 70% to products
    env_share = energy_released * 0.3   # 30% to environment

    # Distribute to products
    {updated_products, total_product_energy} = Enum.reduce(products, {[], 0.0}, fn product, {acc, total_energy} ->
      energy_share_per_product = product_share / num_products
      energy_transfer = AII.Types.Conserved.transfer(
        environment.total_energy,
        product.energy,
        energy_share_per_product
      )

      case energy_transfer do
        {:ok, env_energy, product_energy} ->
          updated_product = %{product | energy: product_energy}
          {[updated_product | acc], total_energy + energy_share_per_product}
        {:error, _} ->
          {acc, total_energy}
      end
    end)

    # Transfer remaining to environment
    remaining_env_energy = env_share + (product_share - total_product_energy)
    env_transfer = AII.Types.Conserved.transfer(
      hd(updated_products).energy,  # Use first product's energy as source
      environment.total_energy,
      remaining_env_energy
    )

    case env_transfer do
      {:ok, _, final_env_energy} ->
        {:ok, Enum.reverse(updated_products), final_env_energy}
      {:error, _} ->
        {:error, :conservation_violation}
    end
  end

  defp can_bind_to_catalyst?(substrate, catalyst) do
    # Check if substrate has available binding sites
    length(catalyst.bound_molecules) < catalyst.active_sites and
    # Simplified binding compatibility
    substrate.molecular_weight < 1000.0  # Size limit
  end

  defp neural_network_predict(catalyst, substrate, environment) do
    # Simplified neural network prediction
    # In real implementation, this would run on NPU

    # Input features
    features = [
      catalyst.catalyst_type == :enzyme,
      catalyst.efficiency,
      substrate.molecular_weight,
      environment.temperature,
      environment.ph
    ]

    # Simple prediction logic
    confidence = cond do
      catalyst.catalyst_type == :enzyme and catalyst.efficiency > 0.8 ->
        0.95
      catalyst.catalyst_type == :metal and environment.temperature > 500.0 ->
        0.85
      substrate.molecular_weight < 100.0 ->
        0.75
      true ->
        0.6
    end

    pathway = case confidence do
      conf when conf > 0.9 -> :optimal
      conf when conf > 0.8 -> :favorable
      conf when conf > 0.7 -> :moderate
      _ -> :unfavorable
    end

    {:ok, pathway, confidence}
  end

  defp transform_substrate(substrate, pathway) do
    # Transform substrate based on reaction pathway
    case pathway do
      :optimal ->
        %{substrate |
          molecule_type: :product,
          molecular_formula: transform_formula(substrate.molecular_formula, "oxidation"),
          temperature: substrate.temperature + 50.0
        }
      :favorable ->
        %{substrate |
          molecule_type: :product,
          molecular_formula: transform_formula(substrate.molecular_formula, "reduction"),
          temperature: substrate.temperature + 25.0
        }
      :moderate ->
        %{substrate |
          molecule_type: :product,
          molecular_formula: transform_formula(substrate.molecular_formula, "substitution"),
          temperature: substrate.temperature + 10.0
        }
      _ ->
        substrate  # No transformation
    end
  end

  defp transform_formula(formula, transformation) do
    # Simplified formula transformation
    case transformation do
      "oxidation" -> String.replace(formula, "H2O", "CO2")
      "reduction" -> String.replace(formula, "CO2", "H2O")
      "substitution" -> String.replace(formula, "H", "F")
      _ -> formula
    end
  end

  defp calculate_diffusion_coefficient(molecule, environment) do
    # Stokes-Einstein relation (simplified)
    base_coefficient = 1.0 / molecule.molecular_weight

    # Temperature dependence
    temp_factor = environment.temperature / 298.0  # Normalized to room temp

    # Viscosity effect (simplified)
    viscosity_factor = 1.0 / (1.0 + environment.ionic_strength * 0.1)

    base_coefficient * temp_factor * viscosity_factor
  end

  defp record_conservation_violation(violation_type, expected, actual, environment) do
    # Record conservation violation for debugging
    violation_record = %{
      timestamp: environment.time || 0.0,
      type: violation_type,
      expected: expected,
      actual: actual,
      difference: calculate_difference(expected, actual)
    }

    # In real implementation, would store this for analysis
    IO.warn("Conservation violation detected: #{inspect(violation_record)}")
  end

  defp calculate_difference(expected, actual) when is_map(expected) and is_map(actual) do
    # Calculate difference between atom maps
    all_elements = Map.keys(expected) ++ Map.keys(actual) |> Enum.uniq()

    Enum.reduce(all_elements, %{}, fn element, acc ->
      expected_count = Map.get(expected, element, 0)
      actual_count = Map.get(actual, element, 0)
      difference = expected_count - actual_count

      if difference != 0 do
        Map.put(acc, element, difference)
      else
        acc
      end
    end)
  end

  @doc """
  Create chemical reaction system
  """
  def create_reaction_system(num_molecules \\ 20) when is_integer(num_molecules) do
    # Create diverse set of molecules
    molecules = for i <- 1..num_molecules do
      molecule_type = case rem(i, 4) do
        0 -> :reactant
        1 -> :product
        2 -> :catalyst
        _ -> :reactant
      end

      formula = case rem(i, 5) do
        0 -> "H2"
        1 -> "O2"
        2 -> "CO2"
        3 -> "H2O"
        _ -> "CH4"
      end

      %{
        molecule_id: i,
        molecule_type: molecule_type,
        molecular_formula: formula,
        molecular_weight: molecular_weight(formula),

        position: {
          (:rand.uniform() - 0.5) * 100,
          (:rand.uniform() - 0.5) * 100,
          (:rand.uniform() - 0.5) * 100
        },
        velocity: {
          (:rand.uniform() - 0.5) * 5,
          (:rand.uniform() - 0.5) * 5,
          (:rand.uniform() - 0.5) * 5
        },
        temperature: 298.0 + (:rand.uniform() - 0.5) * 50,
        concentration: 0.1 + :rand.uniform() * 0.5,
        bonds: generate_bonds(formula),

        mass: AII.Types.Conserved.new(molecular_weight(formula), :initial),
        charge: AII.Types.Conserved.new(net_charge(formula), :initial),
        energy: AII.Types.Conserved.new(100.0, :initial),
        atoms: AII.Types.Conserved.new(count_atoms_by_element(formula), :initial),
        information: AII.Types.Conserved.new(25.0, :initial)
      }
    end

    # Create catalyst
    catalyst = %{
      catalyst_id: num_molecules + 1,
      catalyst_type: :enzyme,
      active_sites: 5,
      activation_energy: 50.0,

      temperature: 310.0,
      bound_molecules: [],
      efficiency: 0.85,

      energy: AII.Types.Conserved.new(200.0, :initial),
      information: AII.Types.Conserved.new(100.0, :initial)
    }

    # Create environment
    environment = %{
      volume: 1000.0,  # Liters
      pressure: 1.0,     # Atmospheres
      temperature: 298.0, # Kelvin
      ph: 7.0,
      ionic_strength: 0.1,
      solvent_concentration: 0.8,

      total_energy: AII.Types.Conserved.new(10000.0, :environment),
      total_mass: AII.Types.Conserved.new(5000.0, :environment),
      total_charge: AII.Types.Conserved.new(0.0, :environment),
      time: 0.0
    }

    %{
      molecules: molecules,
      catalyst: catalyst,
      environment: environment,
      time: 0.0,
      step: 0
    }
  end

  @doc """
  Run chemical reaction simulation
  """
  def run_simulation(initial_state, opts \\ []) do
    steps = Keyword.get(opts, :steps, 1000)
    dt = Keyword.get(opts, :dt, 0.001)

    # AII runtime automatically dispatches to optimal hardware
    AIIRuntime.simulate(initial_state, steps: steps, dt: dt)
  end

  @doc """
  Get reaction system statistics
  """
  def reaction_stats(state) do
    molecules = Map.get(state, :molecules, [])
    num_molecules = length(molecules)

    avg_temp = if num_molecules > 0 do
      molecules
      |> Enum.map(& &1.temperature)
      |> Enum.sum()
      |> Kernel./(num_molecules)
    else
      0.0
    end

    avg_conc = if num_molecules > 0 do
      molecules
      |> Enum.map(& &1.concentration)
      |> Enum.sum()
      |> Kernel./(num_molecules)
    else
      0.0
    end

    %{
      total_molecules: num_molecules,
      reactants: Enum.count(molecules, & &1.molecule_type == :reactant),
      products: Enum.count(molecules, & &1.molecule_type == :product),

      average_temperature: avg_temp,
      average_concentration: avg_conc,

      total_mass: Enum.sum(Enum.map(molecules, & &1.mass.value)),
      total_charge: get_in(state, [:environment, :total_charge, :value]) || 0.0,
      total_energy: get_in(state, [:environment, :total_energy, :value]) || 0.0,
      catalyst_efficiency: get_in(state, [:catalyst, :efficiency]) || 0.0,
      bound_to_catalyst: (get_in(state, [:catalyst, :bound_molecules]) || []) |> length(),
      conservation_status: %{
        atoms: true,  # Placeholder
        charge: true, # Placeholder
        energy: true, # Placeholder
        mass: true    # Placeholder
      }
    }
  end

  # Helper functions for system creation
  def molecular_weight(formula) when is_map(formula) do
    # Handle map format like %{H: 2, O: 1}
    atom_weights = %{
      "H" => 1.008,
      "O" => 15.999,
      "C" => 12.011,
      "N" => 14.007,
      "F" => 18.998
    }

    Enum.reduce(formula, 0.0, fn {element, count}, acc ->
      element_str = to_string(element)
      weight = Map.get(atom_weights, element_str, 12.011)
      acc + weight * count
    end)
  end

  def molecular_weight(formula) when is_binary(formula) do
    # Handle string format like "H2O"
    atom_weights = %{
      "H" => 1.008,
      "O" => 15.999,
      "C" => 12.011,
      "N" => 14.007,
      "F" => 18.998
    }

    Regex.scan(~r/([A-Z][a-z]*)(\d*)/, formula)
    |> Enum.reduce(0.0, fn [_full, element, count_str], acc ->
      atom_count = if count_str == "", do: 1, else: String.to_integer(count_str)
      weight = Map.get(atom_weights, element, 12.011)  # Default to carbon
      acc + weight * atom_count
    end)
  end

  def net_charge(formula) when is_map(formula) do
    # Handle map format like %{"H+": 1, "OH-": 1}
    Enum.reduce(formula, 0, fn {ion, count}, acc ->
      charge = case ion do
        "H+" -> 1
        "OH-" -> -1
        "Na+" -> 1
        "Cl-" -> -1
        _ -> 0  # Default neutral
      end
      acc + charge * count
    end)
  end

  def net_charge(formula) when is_binary(formula) do
    # Simplified net charge calculation for strings
    # In real implementation, would parse oxidation states
    0.0  # Assume neutral molecules
  end

  def generate_bonds(formula) when is_map(formula) do
    # Handle map format like %{H: 4, C: 1}
    atoms = Enum.flat_map(formula, fn {el, cnt} -> List.duplicate(to_string(el), cnt) end)

    Enum.with_index(atoms, fn _element, index ->
      if index < length(atoms) - 1 do
        %{
          type: :covalent,
          strength: 1.0,
          atoms: [index, index + 1]
        }
      end
    end)
    |> Enum.filter(& &1)
  end

  def generate_bonds(formula) when is_binary(formula) do
    # Generate simplified bond structure
    atoms = Regex.scan(~r/([A-Z][a-z]*)(\d*)/, formula)

    Enum.with_index(atoms, fn [_full, element, _count_str], index ->
      if index < length(atoms) - 1 do
        %{
          type: :covalent,
          strength: 1.0,
          atoms: [index, index + 1]
        }
      end
    end)
    |> Enum.filter(& &1)
  end

  defp check_conservation(state, quantity) do
    # Check if quantity is conserved in the system
    case quantity do
      :mass ->
        molecule_mass = Enum.sum(Enum.map(state.molecules, & &1.mass.value))
        catalyst_mass = state.catalyst.mass.value
        env_mass = state.environment.total_mass.value
        total_mass = molecule_mass + catalyst_mass + env_mass

        # Mass should be constant
        abs(total_mass - 5000.0) < 0.001  # Initial total mass

      :charge ->
        molecule_charge = Enum.sum(Enum.map(state.molecules, & &1.charge.value))
        catalyst_charge = 0.0  # Catalyst is neutral
        env_charge = state.environment.total_charge.value
        total_charge = molecule_charge + catalyst_charge + env_charge

        # Charge should be conserved
        abs(total_charge) < 0.001

      :energy ->
        molecule_energy = Enum.sum(Enum.map(state.molecules, & &1.energy.value))
        catalyst_energy = state.catalyst.energy.value
        env_energy = state.environment.total_energy.value
        total_energy = molecule_energy + catalyst_energy + env_energy

        # Energy should be conserved
        abs(total_energy - 10000.0) < 0.001

      :atoms ->
        # Check atom conservation (simplified)
        true  # Would need detailed atom counting
    end
  end
end
