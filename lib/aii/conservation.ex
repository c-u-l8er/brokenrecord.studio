defmodule AII.Conservation do
  @tolerance 0.0001

  # Calculate total information in a data structure
  def total_information(data) when is_map(data) do
    data
    |> Map.values()
    |> Enum.map(&extract_information/1)
    |> Enum.sum()
  end

  defp extract_information(%AII.Types.Conserved{value: v}), do: v

  defp extract_information(%AII.Types.Particle{information: info}),
    do: info.value

  defp extract_information(list) when is_list(list),
    do: Enum.sum(Enum.map(list, &extract_information/1))

  defp extract_information(_), do: 0.0

  # Verify conservation between two states
  def verify(before, after_state, opts \\ []) do
    tolerance = opts[:tolerance] || @tolerance

    before_info = total_information(before)
    after_info = total_information(after_state)

    diff = before_info - after_info

    if after_info > before_info + tolerance do
      raise AII.Types.ConservationViolation, """
      Conservation violation detected:
      Before: #{before_info}
      After: #{after_info}
      Difference: #{diff} (tolerance: #{tolerance})
      Before data: #{inspect(before)}
      After data: #{inspect(after_state)}
      """
    else
      :ok
    end
  end

  # Compile-time symbolic verification (ADVANCED - optional)
  def symbolic_verify(ast) do
    # Analyze AST to prove conservation symbolically
    # This is hard - start with runtime checks
    # Future: constraint solver integration
    :not_implemented
  end
end
