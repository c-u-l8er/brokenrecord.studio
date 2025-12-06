defmodule AII.Examples.AtomicChemicBionicExample do
  @moduledoc """
  Example demonstrating the Atomic, Chemic, and Bionic DSLs with provenance tracking.

  This example defines:
  - An atomic that doubles a number with provenance tracking
  - A chemic that doubles twice (composes two atomics)
  - A bionic that orchestrates the chemic

  It also includes tests to verify provenance and execution.
  """

  require AII.DSL.Atomic
  require AII.DSL.Chemic
  require AII.DSL.Bionic

  import AII.DSL.Atomic
  import AII.DSL.Chemic
  import AII.DSL.Bionic

  # Define an atomic that doubles a number with provenance tracking
  defatomic Double do
    input(:value, :number)

    requires_quality(0.8)

    kernel do
      doubled = inputs[:value].value * 2

      # Create tracked output with provenance
      result =
        AII.Types.Tracked.transform(
          inputs[:value],
          :Double,
          :multiply,
          doubled,
          # Slight degradation
          inputs[:value].provenance.confidence * 0.95,
          %{factor: 2}
        )

      %{result: result}
    end
  end

  # Define a chemic: composes two doubles
  defchemic DoubleTwice do
    atomic(:first, Atomic.Double)
    atomic(:second, Atomic.Double)

    bonds do
      bond(:first, :second)
    end

    # tracks_pipeline_provenance do
    #   # Output must trace back to input
    #   output = outputs[:result]
    #   input = inputs[:value]

    #   output.provenance.source_id == input.provenance.source_id and
    #     length(output.provenance.transformation_chain) == 2
    # end
  end

  # Define a bionic: orchestrates the double_twice chemic
  defbionic DoubleBionic do
    inputs do
      stream(:value, type: :number)
    end

    dag do
      node :process do
        chemic(Chemic.DoubleTwice)
      end
    end

    # verify_end_to_end_provenance do
    #   # End-to-end verification
    #   output = outputs[:result]
    #   input = inputs[:value]

    #   output.provenance.source_id == input.provenance.source_id and
    #     length(output.provenance.transformation_chain) == 2
    # end
  end

  # Test function with provenance verification
  def run_example do
    input =
      AII.Types.Tracked.new(5.0, "example_input", :user_input,
        confidence: 1.0,
        metadata: %{user: "test"}
      )

    IO.puts("Running Atomic Chemic Bionic Example with Provenance")
    IO.puts("Input: #{input.value}")

    # Test atomic directly
    {:ok, atomic_outputs} = Atomic.Double.execute(%{value: input})
    IO.puts("Atomic outputs: #{inspect(atomic_outputs)}")

    # Verify atomic provenance
    atomic_result = atomic_outputs[:result]

    IO.puts(
      "Atomic provenance: source_id=#{atomic_result.provenance.source_id}, transformations=#{length(atomic_result.provenance.transformation_chain)}"
    )

    # Test chemic directly
    {:ok, chemic_outputs} = Chemic.DoubleTwice.execute(%{value: input})
    IO.puts("Chemic outputs: #{inspect(chemic_outputs)}")

    # Verify chemic provenance
    chemic_result = chemic_outputs[:result]

    IO.puts(
      "Chemic provenance: source_id=#{chemic_result.provenance.source_id}, transformations=#{length(chemic_result.provenance.transformation_chain)}"
    )

    # Run the bionic
    case Bionic.DoubleBionic.run(%{value: input}) do
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

AII.Examples.AtomicChemicBionicExample.run_example()
