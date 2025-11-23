defmodule BrokenRecord.Zero.IR do
  @moduledoc """
  Intermediate Representation for interaction nets.

  This is the AST we optimize and transform.
  """

  defstruct [
    agents: [],
    rules: [],
    topology: %{},
    metadata: %{}
  ]

  def lower(agents, rules, module_name \\ nil) do
    metadata = case module_name do
      nil -> %{}
      module -> %{module: module}
    end

    %__MODULE__{
      agents: Enum.map(agents, &lower_agent/1),
      rules: Enum.map(rules, &lower_rule/1),
      metadata: metadata
    }
  end

  def lower_agent(agent) do
    conserves = Map.get(agent, :conserves, [])
    %{
      name: agent.name,
      fields: lower_fields(agent.fields),
      conserves: conserves,
      size: compute_size(agent.fields),
      alignment: compute_alignment(agent.fields)
    }
  end

  def lower_fields(fields) do
    Enum.map(fields, fn field ->
      name = if is_map(field), do: field.name, else: field
      type = if is_map(field), do: field.type, else: field

      %{
        name: name,
        type: lower_type(type),
        offset: 0,  # Computed later
        size: type_size(type)
      }
    end)
  end

  defp lower_type(:vec3), do: {:array, :float32, 3}
  defp lower_type(:float), do: :float32
  defp lower_type(:int), do: :int32
  defp lower_type(t), do: t

  def type_size(:vec3), do: 12
  def type_size(:float), do: 4
  def type_size(:int), do: 4
  def type_size(_), do: 8

  defp compute_size(fields) do
    fields
    |> Enum.map(fn field ->
      type = if is_map(field), do: field.type, else: field
      type_size(type)
    end)
    |> Enum.sum()
  end

  defp compute_alignment(fields) do
    fields
    |> Enum.map(fn field ->
      type = if is_map(field), do: field.type, else: field
      type_size(type)
    end)
    |> Enum.max()
    |> max(16)  # Minimum 16-byte alignment for SIMD
  end

  def lower_rule(rule) do
    %{
      name: rule.name,
      params: rule.params,
      body: lower_body(rule.body),
      metadata: %{
        parallel: analyze_parallelism(rule.body),
        vectorizable: analyze_vectorization(rule.body)
      }
    }
  end

  defp lower_body(body) do
    # Transform high-level operations to IR operations
    # This is where we normalize the computation
    body  # Simplified for now
  end

  defp analyze_parallelism(_body) do
    # Analyze if rule can be parallelized
    :data_parallel  # or :embarrassingly_parallel, :sequential
  end

  defp analyze_vectorization(_body) do
    # Check if SIMD can be applied
    true
  end
end