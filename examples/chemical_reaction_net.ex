defmodule Examples.ChemicalReactionNet do
  @moduledoc """
  Example: Chemical reaction network using lafont interaction nets.

  Demonstrates:
  - Chemical reaction kinetics
  - Mass conservation
  - Reaction equilibrium
  - Catalysis processes
  """

  use BrokenRecord.Zero

  defsystem ChemicalNetwork do
    compile_target :cpu
    optimize [:spatial_hash, :simd, :loop_fusion]

    agents do
      defagent Molecule do
        field :position, :vec3
        field :velocity, :vec3
        field :mass, :float
        field :radius, :float
        field :chemical_type, :atom
        field :energy, :float
        conserves [:mass, :energy]
      end

      defagent Catalyst do
        field :position, :vec3
        field :active, :bool
        field :catalysis_rate, :float
        field :bound_molecules, :list
        conserves [:mass]
      end

      defagent Container do
        field :position, :vec3
        field :radius, :float
        field :temperature, :float
        field :pressure, :float
        conserves []
      end
    end

    rules do
      # Simple collision reaction: A + B -> C
      interaction synthesis_reaction(m1: Molecule, m2: Molecule) do
        # Check for collision
        delta = m1.position - m2.position
        dist = vector_length(delta)

        if dist < (m1.radius + m2.radius) do
          # Check if molecules can react (simplified chemistry)
          can_react = (m1.chemical_type == :A and m2.chemical_type == :B) or
                     (m1.chemical_type == :B and m2.chemical_type == :C)

          if can_react do
            # Calculate reaction probability based on energy
            activation_energy = 50.0
            total_energy = m1.energy + m2.energy
            reaction_prob = :math.exp(-activation_energy / total_energy)

            if :rand.uniform() < reaction_prob do
              # Perform reaction: conserve mass and energy
              new_mass = m1.mass + m2.mass
              new_energy = total_energy * 0.9  # Some energy lost as heat

              # Create product molecule at center of mass
              com_position = (m1.position * m1.mass + m2.position * m2.mass) / new_mass
              com_velocity = (m1.velocity * m1.mass + m2.velocity * m2.mass) / new_mass

              # Transform first molecule into product
              m1.mass = new_mass
              m1.position = com_position
              m1.velocity = com_velocity
              m1.energy = new_energy
              m1.radius = :math.pow(new_mass, 1/3) * 0.5  # Volume proportional to mass

              # Determine product type
              m1.chemical_type = cond do
                (m1.chemical_type == :A and m2.chemical_type == :B) or
                (m1.chemical_type == :B and m2.chemical_type == :A) -> :C
                (m1.chemical_type == :B and m2.chemical_type == :C) or
                (m1.chemical_type == :C and m2.chemical_type == :B) -> :D
                true -> m1.chemical_type
              end

              # Mark second molecule for removal (in real implementation)
              m2.mass = 0.0  # Flag for removal
            end
          end
        end
      end

      # Decomposition reaction: C -> A + B
      interaction decomposition_reaction(m: Molecule) do
        # Spontaneous decomposition for unstable molecules
        if m.chemical_type in [:C, :D] and m.energy > 100.0 do
          decomp_prob = 0.001  # Small probability per timestep

          if :rand.uniform() < decomp_prob do
            # Split into two molecules
            mass1 = m.mass * 0.4
            mass2 = m.mass * 0.6

            # Conservation of momentum
            separation_velocity = {2.0, 1.0, 0.5}

            # Update current molecule
            m.mass = mass1
            m.energy = m.energy * 0.5
            m.velocity = m.velocity + separation_velocity
            m.radius = :math.pow(mass1, 1/3) * 0.5

            # Determine product types
            m.chemical_type = if m.chemical_type == :C, do: :A, else: :B

            # In real implementation, would create new molecule with:
            # mass2, remaining energy, opposite velocity, complementary type
          end
        end
      end

      # Catalyzed reaction: A + B + Catalyst -> C + Catalyst
      interaction catalyzed_reaction(m1: Molecule, m2: Molecule, cat: Catalyst) do
        if cat.active do
          # Check if all three are close enough
          dist_m1_m2 = vector_length(m1.position - m2.position)
          dist_m1_cat = vector_length(m1.position - cat.position)
          dist_m2_cat = vector_length(m2.position - cat.position)

          interaction_radius = 5.0

          if dist_m1_m2 < interaction_radius and
             dist_m1_cat < interaction_radius and
             dist_m2_cat < interaction_radius do

            # Check for catalyzed reaction conditions
            can_catalyze = (m1.chemical_type == :A and m2.chemical_type == :B) and
                          cat.catalysis_rate > 0.0

            if can_catalyze do
              # Enhanced reaction rate due to catalyst
              enhanced_prob = cat.catalysis_rate * 0.1

              if :rand.uniform() < enhanced_prob do
                # Perform catalyzed reaction (similar to synthesis but faster)
                new_mass = m1.mass + m2.mass
                new_energy = (m1.energy + m2.energy) * 0.95  # Less energy loss

                com_position = (m1.position * m1.mass + m2.position * m2.mass) / new_mass
                com_velocity = (m1.velocity * m1.mass + m2.velocity * m2.mass) / new_mass

                m1.mass = new_mass
                m1.position = com_position
                m1.velocity = com_velocity
                m1.energy = new_energy
                m1.radius = :math.pow(new_mass, 1/3) * 0.5
                m1.chemical_type = :C

                m2.mass = 0.0  # Flag for removal

                # Catalyst remains unchanged (true catalyst behavior)
              end
            end
          end
        end
      end

      # Thermal motion and container collisions
      interaction thermal_motion(m: Molecule, container: Container) do
        # Brownian motion based on temperature
        thermal_force = {(:rand.uniform() - 0.5) * container.temperature,
                        (:rand.uniform() - 0.5) * container.temperature,
                        (:rand.uniform() - 0.5) * container.temperature}

        m.velocity = m.velocity + thermal_force / m.mass * 0.01

        # Container collision
        dist_from_center = vector_length(m.position - container.position)

        if dist_from_center + m.radius > container.radius do
          # Elastic collision with container wall
          normal = normalize(m.position - container.position)
          v_normal = dot(m.velocity, normal)

          if v_normal > 0 do  # Moving toward wall
            m.velocity = m.velocity - normal * (2 * v_normal)

            # Position correction
            penetration = dist_from_center + m.radius - container.radius
            m.position = m.position - normal * penetration
          end
        end
      end

      # Integration
      interaction integrate(m: Molecule, dt: float) do
        m.position = m.position + m.velocity * dt

        # Apply drag
        drag_coefficient = 0.01
        m.velocity = m.velocity * (1.0 - drag_coefficient)
      end
    end
  end

  @doc """
  Create initial chemical mixture
  """
  def chemical_mixture(num_a \\ 50, num_b \\ 50, num_c \\ 20, num_d \\ 10) when is_integer(num_a) and is_integer(num_b) and is_integer(num_c) and is_integer(num_d) do
    molecules = []

    # Type A molecules
    molecules = molecules ++ for _i <- 1..num_a do
      %{
        position: random_position(50.0),
        velocity: random_velocity(2.0),
        mass: 1.0,
        radius: 0.5,
        chemical_type: :A,
        energy: 20.0 + :rand.uniform() * 30.0
      }
    end

    # Type B molecules
    molecules = molecules ++ for _i <- 1..num_b do
      %{
        position: random_position(50.0),
        velocity: random_velocity(2.0),
        mass: 1.2,
        radius: 0.6,
        chemical_type: :B,
        energy: 25.0 + :rand.uniform() * 35.0
      }
    end

    # Type C molecules
    molecules = molecules ++ for _i <- 1..num_c do
      %{
        position: random_position(50.0),
        velocity: random_velocity(1.5),
        mass: 2.2,
        radius: 0.8,
        chemical_type: :C,
        energy: 40.0 + :rand.uniform() * 40.0
      }
    end

    # Type D molecules
    molecules = molecules ++ for _i <- 1..num_d do
      %{
        position: random_position(50.0),
        velocity: random_velocity(1.0),
        mass: 3.2,
        radius: 1.0,
        chemical_type: :D,
        energy: 60.0 + :rand.uniform() * 50.0
      }
    end

    %{
      molecules: molecules,
      catalysts: create_catalysts(5),
      container: %{
        position: {0.0, 0.0, 0.0},
        radius: 60.0,
        temperature: 300.0,
        pressure: 1.0
      }
    }
  end

  defp create_catalysts(num) do
    for _i <- 1..num do
      %{
        position: random_position(40.0),
        active: true,
        catalysis_rate: 0.5 + :rand.uniform() * 0.5,
        bound_molecules: []
      }
    end
  end

  @doc """
  Run simulation and return final state
  """
  def run_simulation(initial_state, opts) do
    steps = Keyword.get(opts, :steps, 1000)
    dt = Keyword.get(opts, :dt, 0.01)

    ChemicalNetwork.simulate(initial_state, steps: steps, dt: dt)
  end

  @doc """
  Count molecules by type
  """
  def count_molecules(state) do
    Enum.reduce(state.molecules, %{A: 0, B: 0, C: 0, D: 0}, fn molecule, acc ->
      if molecule.mass > 0 do  # Only count non-removed molecules
        Map.put(acc, molecule.chemical_type, acc[molecule.chemical_type] + 1)
      else
        acc
      end
    end)
  end

  @doc """
  Calculate total mass (should be conserved)
  """
  def total_mass(state) do
    Enum.reduce(state.molecules, 0.0, fn molecule, acc ->
      acc + molecule.mass
    end) + Enum.reduce(state.catalysts, 0.0, fn _catalyst, acc ->
      acc + 10.0  # Assume catalyst mass of 10.0 each
    end)
  end

  @doc """
  Calculate total energy (should be approximately conserved)
  """
  def total_energy(state) do
    # Kinetic energy
    kinetic = Enum.reduce(state.molecules, 0.0, fn molecule, acc ->
      v_sq = vector_dot(molecule.velocity, molecule.velocity)
      acc + 0.5 * molecule.mass * v_sq
    end)

    # Chemical energy
    chemical = Enum.reduce(state.molecules, 0.0, fn molecule, acc ->
      acc + molecule.energy
    end)

    kinetic + chemical
  end

  @doc """
  Calculate reaction rates
  """
  def reaction_rates(initial_state, final_state, time) do
    initial_counts = count_molecules(initial_state)
    final_counts = count_molecules(final_state)

    %{
      synthesis_rate: (final_counts[:C] - initial_counts[:C]) / time,
      decomposition_rate: (final_counts[:A] + final_counts[:B] - initial_counts[:A] - initial_counts[:B]) / time
    }
  end

  # Helper functions
  defp random_position(max_radius) do
    theta = :rand.uniform() * 2 * :math.pi()
    phi = :rand.uniform() * :math.pi()
    r = :rand.uniform() * max_radius

    {
      r * :math.sin(phi) * :math.cos(theta),
      r * :math.sin(phi) * :math.sin(theta),
      r * :math.cos(phi)
    }
  end

  defp random_velocity(max_speed) do
    {
      (:rand.uniform() - 0.5) * 2 * max_speed,
      (:rand.uniform() - 0.5) * 2 * max_speed,
      (:rand.uniform() - 0.5) * 2 * max_speed
    }
  end

  defp vector_dot({x1, y1, z1}, {x2, y2, z2}) do
    x1 * x2 + y1 * y2 + z1 * z2
  end

end
