defmodule BrokenRecord.Zero.DSL do
  @moduledoc "High-level DSL for defining physics systems"

  defmacro defsystem(name, do: block) do
    quote do
      defmodule unquote(name) do
        use BrokenRecord.Zero
        unquote(block)
      end
    end
  end

  defmacro agents(do: block) do
    quote do
      unquote(block)
    end
  end

  defmacro defagent(name, do: block) do
    # Parse agent definition
    {fields, conserves} = parse_agent_block(block)

    agent_def = %{
      name: name,
      fields: fields,
      conserves: conserves,
      metadata: %{}
    }

    quote do
      Module.put_attribute(__MODULE__, :agents, unquote(Macro.escape(agent_def)))
    end
  end

  defmacro field(name, type) do
    quote do
      {unquote(name), unquote(type)}
    end
  end

  defmacro conserves(quantities) do
    quote do
      {:conserves, unquote(quantities)}
    end
  end

  defmacro rules(do: block) do
    quote do
      unquote(block)
    end
  end

  defmacro interaction(signature, do: block) do
    # Parse interaction rule
    {rule_name, params} = parse_signature(signature)
    body = parse_body(block)

    rule_def = %{
      name: rule_name,
      params: params,
      body: body,
      metadata: %{}
    }

    quote do
      Module.put_attribute(__MODULE__, :rules, unquote(Macro.escape(rule_def)))
    end
  end

  defmacro compile_target(target) do
    quote do
      Module.put_attribute(__MODULE__, :compile_opts,
        Keyword.put(@compile_opts || [], :target, unquote(target)))
    end
  end

  defmacro optimize(passes) do
    quote do
      Module.put_attribute(__MODULE__, :compile_opts,
        Keyword.put(@compile_opts || [], :optimize, unquote(passes)))
    end
  end

  # Helper to parse agent blocks
  defp parse_agent_block({:__block__, _, statements}) do
    fields = for {:field, _, [name, type]} <- statements, do: {name, type}
    conserves = for {:conserves, _, [list]} <- statements, do: list
    {fields, List.flatten(conserves)}
  end
  defp parse_agent_block(_), do: {[], []}

  defp parse_signature({:call, _, [{name, _, params}]}) when is_list(params) do
    parsed_params = Enum.map(params, fn
      {:"::", _, [var, type]} -> {var, type}
      var -> {var, :any}
    end)
    {name, parsed_params}
  end

  defp parse_signature({name, _, params}) when is_list(params) do
    parsed_params = Enum.map(params, fn
      {:"::", _, [var, type]} -> {var, type}
      var -> {var, :any}
    end)
    {name, parsed_params}
  end

  defp parse_body(block), do: block
end