defmodule AII.Examples.AtomicChemicBionicExample do
  @moduledoc """
  Example demonstrating the Atomic, Chemic, and Bionic DSLs with provenance tracking.

  This example defines:
  - An atomic that doubles a number with provenance tracking
  - A chemic that doubles twice (composes two atomics)
  - A bionic that orchestrates the chemic

  It also includes tests to verify provenance and execution.
  """

  use AII

  # Define an atomic that doubles a number with provenance tracking
  defatomic Double do
    input(:value)

    transform do
      new_value = inputs[:value].value * 2
      conserved = %{inputs[:value] | value: new_value}
      AII.Types.Conserved.transform(conserved, :multiply, %{factor: 2})
    end

    kernel do
      result = transform_function(atomic_state, inputs)
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

  # Test function with provenance verification
  def run_example do
    input = AII.Types.Conserved.new(5.0, :user_input, source_id: "example_input", confidence: 1.0)

    IO.puts("Running Atomic Chemic Bionic Example with Provenance")
    IO.puts("Input: #{input.value}")

    # # Test atomic directly
    # IO.puts("Testing atomic directly...")
    # {:ok, _, atomic_outputs} = Atomical.Double.execute(%{}, %{value: input})
    # IO.puts("Atomic outputs: #{inspect(atomic_outputs)}")

    # # Verify atomic provenance
    # atomic_result = atomic_outputs[:result]

    # IO.puts(
    #   "Atomic provenance: source_id=#{atomic_result.provenance.source_id}, transformations=#{length(atomic_result.provenance.transformation_chain)}"
    # )
    # )

    # Test chemic directly
    IO.puts("Testing chemic directly...")
    IO.puts("Chemic metadata: #{inspect(Chemical.DoubleTwice.__chemic_metadata__())}")
    chemic_state = %{atomics: %{first: %{}, second: %{}}}
    {:ok, _, chemic_outputs} = Chemical.DoubleTwice.execute(chemic_state, %{value: input})
    IO.puts("Chemic outputs: #{inspect(chemic_outputs)}")

    # Verify chemic provenance
    chemic_result = chemic_outputs[:result]

    IO.puts(
      "Chemic provenance: source_id=#{chemic_result.provenance.source_id}, transformations=#{length(chemic_result.provenance.transformation_chain)}"
    )

    # Run the bionic
    case Bionical.DoubleBionic.run(%{value: input}) do
      {:ok, outputs} ->
        IO.puts("Bionic outputs: #{inspect(outputs)}")
        result = outputs[:result]
        IO.puts("Result: #{inspect(result)}")

        if result do
          IO.puts("Output value: #{result.value}")
          IO.puts("Expected: 20.0 (5 * 2 * 2)")

          # Verify bionic provenance
          IO.puts(
            "Bionic provenance: source_id=#{result.provenance.source_id}, transformations=#{length(result.provenance.transformation_chain)}, confidence=#{result.provenance.confidence}"
          )
        else
          IO.puts("✗ Result is nil")
        end
    end
  end
end
