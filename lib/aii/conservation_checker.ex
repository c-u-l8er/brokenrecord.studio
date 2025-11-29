defmodule AII.ConservationChecker do
  @moduledoc """
  Compile-time verification of conservation laws.
  Analyzes interaction AST to prove conservation through symbolic analysis.
  """

  use Agent

  @type conservation_result :: :ok | {:needs_runtime_check, symbolic_expr(), symbolic_expr()}
  @type conserved_quantity :: atom()
  @type symbolic_expr :: {:sum, conserved_quantity} | {:var, atom()} | {:const, number()} |
                         {:add, symbolic_expr(), symbolic_expr()} |
                         {:sub, symbolic_expr(), symbolic_expr()} |
                         {:mul, symbolic_expr(), symbolic_expr()}

  # Start the cache agent
  def start_link(_opts \\ []) do
    Agent.start_link(fn -> %{} end, name: __MODULE__)
  end

  @doc """
  Verify conservation laws for an interaction against agent definitions.

  Returns :ok if all conserved quantities are proven to be conserved,
  or {:needs_runtime_check, before_expr, after_expr} if runtime verification is needed.
  """
  @spec verify(map(), [map()]) :: conservation_result()
  def verify(interaction, agent_defs) do
    # Create cache key from interaction and agent definitions
    cache_key = {interaction, agent_defs}

    # Check cache first
    case get_cached_result(cache_key) do
      {:ok, result} -> result
      :not_found ->
        # Compute result
        result = verify_uncached(interaction, agent_defs)
        # Cache the result
        cache_result(cache_key, result)
        result
    end
  end

  @doc """
  Clear the conservation verification cache.
  """
  @spec clear_cache() :: :ok
  def clear_cache do
    Agent.update(__MODULE__, fn _ -> %{} end)
  end

  # Private functions

  @spec verify_uncached(map(), [map()]) :: conservation_result()
  defp verify_uncached(interaction, agent_defs) do
    # Extract conserved quantities from agents
    conserved = extract_conserved_quantities(agent_defs)

    # Track each quantity through interaction
    Enum.reduce(conserved, :ok, fn quantity, acc ->
      case track_conservation(interaction, quantity) do
        :ok -> acc
        {:needs_runtime_check, before_expr, after_expr} ->
          # For now, accept runtime checks - could be enhanced to combine them
          {:needs_runtime_check, before_expr, after_expr}
      end
    end)
  end

  @spec get_cached_result(term()) :: {:ok, conservation_result()} | :not_found
  defp get_cached_result(cache_key) do
    try do
      case Agent.get(__MODULE__, &Map.get(&1, cache_key)) do
        nil -> :not_found
        result -> {:ok, result}
      end
    catch
      :exit, _ -> :not_found  # Agent not started
    end
  end

  @spec cache_result(term(), conservation_result()) :: :ok
  defp cache_result(cache_key, result) do
    try do
      Agent.update(__MODULE__, &Map.put(&1, cache_key, result))
    catch
      :exit, _ -> :ok  # Agent not started, skip caching
    end
  end

  @doc """
  Track conservation of a specific quantity through an interaction.
  """
  @spec track_conservation(map(), conserved_quantity()) :: conservation_result()
  def track_conservation(interaction, quantity) do
    # Build symbolic expressions for total quantity before and after
    {before_expr, after_expr} = build_expressions(interaction, quantity)

    # Check if expressions are symbolically equal
    if symbolically_equal?(before_expr, after_expr) do
      :ok
    else
      # Cannot prove conservation at compile time
      {:needs_runtime_check, before_expr, after_expr}
    end
  end

  @doc """
  Build symbolic expressions representing total conserved quantity before and after interaction.
  """
  @spec build_expressions(map(), conserved_quantity()) :: {symbolic_expr(), symbolic_expr()}
  def build_expressions(interaction, quantity) do
    # Get all particles involved in the interaction
    particles = extract_particles(interaction)

    # Build expression for total quantity before interaction
    before_expr = build_total_expr(particles, quantity, :before)

    # Build expression for total quantity after interaction
    after_expr = build_total_expr(particles, quantity, :after)

    {before_expr, after_expr}
  end

  @doc """
  Check if two symbolic expressions are algebraically equal.
  """
  @spec symbolically_equal?(symbolic_expr(), symbolic_expr()) :: boolean()
  def symbolically_equal?(expr1, expr2) do
    # Normalize both expressions and compare
    normalize(expr1) == normalize(expr2)
  end

  # Private functions

  defp extract_conserved_quantities(agent_defs) do
    # Extract conserved quantities from agent definitions
    Enum.flat_map(agent_defs, fn agent_def ->
      # Look for :conserves attribute or similar
      Map.get(agent_def, :conserves, [])
    end)
    |> Enum.uniq()
    |> Enum.filter(&is_atom/1)
  end

  defp extract_particles(interaction) do
    # Extract particle variables from interaction AST
    # This is a simplified implementation - would need proper AST walking
    case interaction do
      %{body: {:let, _, [{particle_var, _, _} | _]}} ->
        [particle_var]
      _ ->
        # Default: assume single particle named 'particle'
        [:particle]
    end
  end

  defp build_total_expr(particles, quantity, phase) do
    # Build sum expression for all particles
    particle_exprs = Enum.map(particles, fn particle ->
      build_particle_expr(particle, quantity, phase)
    end)

    # Sum them up
    Enum.reduce(particle_exprs, {:const, 0}, fn expr, acc ->
      {:add, acc, expr}
    end)
  end

  defp build_particle_expr(particle, quantity, phase) do
    # Build expression for quantity in a single particle
    # This would analyze the particle's fields and how they change

    # Simplified: assume quantity is stored in a field with the same name
    field_name = quantity

    case phase do
      :before ->
        {:var, particle, field_name, :before}
      :after ->
        # For now, assume no changes - would need AST analysis
        {:var, particle, field_name, :after}
    end
  end

  @doc """
  Normalize a symbolic expression by applying algebraic simplifications.
  """
  @spec normalize(symbolic_expr()) :: symbolic_expr()
  def normalize(expr) do
    expr
    |> expand()
    |> combine_like_terms()
    |> sort_terms()
  end

  # Expand expressions (e.g., distribute multiplication)
  defp expand({:add, left, right}) do
    {:add, expand(left), expand(right)}
  end
  defp expand({:sub, left, right}) do
    {:sub, expand(left), expand(right)}
  end
  defp expand({:mul, left, right}) do
    {:mul, expand(left), expand(right)}
  end
  defp expand(expr), do: expr

  # Combine like terms (e.g., x + x = 2x)
  defp combine_like_terms(expr) do
    # Simplified implementation - would need proper term collection
    expr
  end

  # Sort terms for canonical representation
  defp sort_terms(expr) do
    # Simplified - sort variables alphabetically
    expr
  end

  @doc """
  Pretty print a symbolic expression for debugging.
  """
  @spec expr_to_string(symbolic_expr()) :: String.t()
  def expr_to_string({:sum, quantity}), do: "Σ#{quantity}"
  def expr_to_string({:var, name}), do: "#{name}"
  def expr_to_string({:var, particle, field, phase}), do: "#{particle}.#{field}[#{phase}]"
  def expr_to_string({:const, value}), do: "#{value}"
  def expr_to_string({:add, left, right}), do: "(#{expr_to_string(left)} + #{expr_to_string(right)})"
  def expr_to_string({:sub, left, right}), do: "(#{expr_to_string(left)} - #{expr_to_string(right)})"
  def expr_to_string({:mul, left, right}), do: "(#{expr_to_string(left)} * #{expr_to_string(right)})"
  def expr_to_string(expr), do: inspect(expr)
end
