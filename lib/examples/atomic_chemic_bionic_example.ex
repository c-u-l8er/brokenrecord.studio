defmodule AII.Examples.AtomicChemicBionicExample do
  @moduledoc """
  Example demonstrating the Atomic, Chemic, and Bionic DSLs.

  This example defines:
  - An atomic that doubles a number
  - A chemic that doubles twice (composes two atomics)
  - A bionic that orchestrates the chemic

  It also includes tests to verify conservation and execution.
  """

  use AII

  # Define an atomic: doubles a number
  defatomic Double do
    input(:value)

    kernel do
      result = AII.Types.Conserved.new(inputs.value.value * 2, :computed)
      %{result: result}
    end
  end

  # Define a chemic: composes two doubles
  defchemic DoubleTwice do
    composition do
      atomic(:first, Atomical.Double)
      atomic(:second, Atomical.Double)

      bonds do
        first -> second
      end
    end
  end

  # Define a bionic: orchestrates the double_twice chemic
  defbionic DoubleBionic do
    dag do
      node Process do
        vertex(Chemical.DoubleTwice)
      end
    end
  end

  # Test function
  def run_example do
    input = AII.Types.Conserved.new(5.0, :user_input)

    IO.puts("Running Atomic Chemic Bionic Example")
    IO.puts("Input: #{input.value}")

    # Test atomic directly
    IO.puts("Testing atomic directly...")
    {:ok, _, atomic_outputs} = Atomical.Double.execute(%{}, %{value: input})
    IO.puts("Atomic outputs: #{inspect(atomic_outputs)}")

    # Test chemic directly
    IO.puts("Testing chemic directly...")
    IO.puts("Chemic metadata: #{inspect(Chemical.DoubleTwice.__chemic_metadata__())}")
    chemic_state = %{atomics: %{first: %{}, second: %{}}}
    {:ok, _, chemic_outputs} = Chemical.DoubleTwice.execute(chemic_state, %{value: input})
    IO.puts("Chemic outputs: #{inspect(chemic_outputs)}")

    # Run the bionic
    case Bionical.DoubleBionic.run(%{value: input}) do
      {:ok, outputs} ->
        IO.puts("Bionic outputs: #{inspect(outputs)}")
        result = outputs[:result]
        IO.puts("Result: #{inspect(result)}")

        if result do
          IO.puts("Output value: #{result.value}")
          IO.puts("Expected: 20.0 (5 * 2 * 2)")
        else
          IO.puts("✗ Result is nil")
        end
    end
  end
end
