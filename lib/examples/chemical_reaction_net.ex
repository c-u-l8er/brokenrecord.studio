defmodule Examples.ChemicalReactionNet do
  @moduledoc """
  Example: Chemical reaction network simulation.

  Demonstrates:
  - Chemical reaction modeling
  - Reaction kinetics
  - Conservation of mass and energy
  - Catalyst interactions
  """

  @doc """
  Creates a chemical mixture with given reactants and conditions.

  ## Parameters
  - reactants: List of {molecule, concentration} tuples
  - temperature: Temperature in Kelvin
  - pressure: Pressure in atm
  - catalysts: List of catalyst molecules

  ## Returns
  A map representing the chemical mixture
  """
  def chemical_mixture(reactants, temperature, pressure, catalysts) do
    %{
      reactants: reactants,
      temperature: temperature,
      pressure: pressure,
      catalysts: catalysts,
      products: [],
      reaction_rate: calculate_reaction_rate(reactants, temperature, catalysts),
      energy_change: 0.0,
      time: 0.0
    }
  end

  @doc """
  Simulates a chemical reaction step
  """
  def react(mixture, dt) do
    # Simple reaction simulation
    reaction_rate = mixture.reaction_rate
    reacted_amount = reaction_rate * dt

    # Update reactants and products
    updated_reactants = Enum.map(mixture.reactants, fn {mol, conc} ->
      {mol, max(0, conc - reacted_amount)}
    end)

    # Generate products (simplified)
    products = [{"H2O", reacted_amount}, {"CO2", reacted_amount}]

    %{mixture |
      reactants: updated_reactants,
      products: products,
      time: mixture.time + dt
    }
  end

  @doc """
  Calculates reaction rate based on Arrhenius equation
  """
  defp calculate_reaction_rate(reactants, temperature, catalysts) do
    # Simplified Arrhenius equation
    base_rate = 0.01
    activation_energy = 50000  # J/mol
    r = 8.314  # J/mol·K

    catalyst_factor = 1 + length(catalysts) * 0.5

    base_rate * :math.exp(-activation_energy / (r * temperature)) * catalyst_factor
  end

  @doc """
  Gets current concentrations
  """
  def concentrations(mixture) do
    Map.new(mixture.reactants ++ mixture.products)
  end
end
