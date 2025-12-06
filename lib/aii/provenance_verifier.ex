defmodule AII.ProvenanceVerifier do
  @moduledoc "Verify data provenance instead of strict conservation"

  require Logger

  def verify_execution(inputs, outputs) do
    # Check that all outputs have valid provenance chains
    Enum.each(outputs, fn {key, value} ->
      verify_output_provenance(value, key)
    end)

    # Ensure transformations are reasonable
    verify_transformation_chain(inputs, outputs)
    :ok
  end

  defp verify_output_provenance(%AII.Types.Conserved{} = conserved, output_key) do
    provenance = conserved.provenance

    # Must have source
    if provenance.source_id == "" do
      raise "Output #{output_key} has no provenance source"
    end

    # Must have reasonable confidence
    if provenance.confidence < 0.1 do
      Logger.warning("Very low confidence output: #{output_key}")
    end

    # Check transformation chain isn't too long (prevents infinite loops)
    if length(provenance.transformation_chain) > 100 do
      raise "Excessive transformation chain in #{output_key}"
    end
  end

  defp verify_output_provenance(_other, _output_key), do: :ok

  defp verify_transformation_chain(inputs, outputs) do
    # Ensure outputs are derived from inputs through valid transformations
    input_sources = extract_sources(inputs)
    output_sources = extract_sources(outputs)

    # All output sources should trace back to input sources
    orphaned_sources = output_sources -- input_sources

    if orphaned_sources != [] do
      Logger.warning("Outputs have sources not present in inputs: #{inspect(orphaned_sources)}")
    end
  end

  defp extract_sources(data) do
    Enum.flat_map(data, fn {_key, value} ->
      case value do
        %AII.Types.Conserved{provenance: %{source_id: source}} -> [source]
        _ -> []
      end
    end)
    |> Enum.uniq()
  end
end
