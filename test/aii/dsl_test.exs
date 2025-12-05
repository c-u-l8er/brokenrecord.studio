defmodule AII.DSLTest do
  use ExUnit.Case
  alias AII.DSL

  defmodule TestAgent do
    use AII.DSL

    Module.register_attribute(__MODULE__, :conserved_quantities, accumulate: true)

    conserved_quantity(:energy, type: :scalar, law: :sum)
    conserved_quantity(:momentum, type: :vector3, law: :sum)

    def __conserved_quantities__, do: @conserved_quantities |> Enum.reverse()

    defagent Particle do
      property(:mass, Float, invariant: true)
      state(:position, AII.Types.Vec3)
      state(:velocity, AII.Types.Vec3)

      derives :kinetic_energy, AII.Types.Energy do
        0.5 * mass * AII.Types.Vec3.magnitude(velocity) ** 2
      end

      conserves(:energy, :momentum)
    end
  end

  defmodule TestInteractions do
    use AII.DSL

    definteraction :test_interaction, accelerator: :auto do
      let particle do
        # Simple test interaction
        particle.velocity = AII.Types.Vec3.add(particle.velocity, {1.0, 0.0, 0.0})
      end
    end
  end

  test "defagent creates module with fields" do
    fields = TestAgent.Particle.__fields__()
    assert length(fields) == 4

    mass_field = Enum.find(fields, &(&1.name == :mass))
    assert mass_field.invariant == true
    assert mass_field.kind == :property

    pos_field = Enum.find(fields, &(&1.name == :position))
    assert pos_field.invariant == false
    assert pos_field.kind == :state

    vel_field = Enum.find(fields, &(&1.name == :velocity))
    assert vel_field.invariant == false
    assert vel_field.kind == :state

    energy_field = Enum.find(fields, &(&1.name == :kinetic_energy))
    assert energy_field.kind == :derived
  end

  # test "conserved_quantity declarations stored" do
  #   quantities = TestAgent.__conserved_quantities__()
  #   assert length(quantities) == 2
  #
  #   energy = Enum.find(quantities, &(&1.name == :energy))
  #   assert energy.type == :scalar
  #   assert energy.law == :sum
  # end
end
