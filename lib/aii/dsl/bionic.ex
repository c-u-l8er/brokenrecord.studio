defmodule AII.DSL.Bionic do
  defmacro defbionic(name, _opts \\ [], do: block) do
    quote do
      defmodule Bionical.unquote(name) do
        use AII.Bionic

        @bionic_name __MODULE__ |> Atom.to_string() |> String.replace("Elixir.Bionical.", "")

        unquote(block)

        @tolerance 0.0001

        Module.register_attribute(__MODULE__, :nodes, accumulate: true)
        Module.register_attribute(__MODULE__, :edges, [])
        Module.register_attribute(__MODULE__, :dag_block, [])

        defp build_execution_plan(metadata) do
          # Parse DAG structure
          dag = parse_dag(metadata.dag_block, metadata.nodes, metadata.edges)

          # Topological sort
          plan = AII.Graph.topological_sort(dag)
          {plan, dag}
        end

        defp execute_plan(plan, dag, state, inputs, opts, metadata) do
          Enum.reduce(plan, {state, inputs}, fn node_name, {st, data} ->
            dependencies = Map.get(dag, node_name, [])
            # Execute node (chemic)
            {updated_st, node_outputs} =
              execute_bionic_node(st, data, node_name, opts, metadata, dependencies)

            # Update data flow
            updated_data = Map.put(data, node_name, node_outputs)

            # Log if tracing
            if opts[:trace] do
              log_execution(node_name, data, node_outputs)
            end

            {updated_st, updated_data}
          end)
        end

        defp execute_bionic_node(state, data, node_name, opts, metadata, dependencies) do
          try do
            # Get node configuration
            {^node_name, node_config} =
              Enum.find(metadata.nodes, fn {n, _} -> n == node_name end)

            # Extract chemic from node config
            chemic_module = extract_chemic_module_from_config(node_config)

            # Get chemic state
            chemic_state = Map.get(state.chemics, node_name, %{atomics: %{}})

            # Get inputs for this node
            node_inputs = gather_node_inputs(data, dependencies)

            # Execute chemic
            {:ok, updated_chemic, outputs} =
              chemic_module.execute(chemic_state, node_inputs)

            # Update state
            updated_state =
              put_in(state.chemics[node_name], updated_chemic)

            {updated_state, outputs}
          rescue
            error ->
              Logger.error("Error executing bionic node #{node_name}: #{inspect(error)}")
              reraise error, __STACKTRACE__
          end
        end

        defp initialize_bionic_state(metadata) do
          # Initialize bionic state with chemics
          chemic_states =
            Enum.reduce(metadata.nodes || [], %{}, fn {node_name, node_config}, acc ->
              chemic_module = extract_chemic_module_from_config(node_config)
              # Initialize chemic state
              chemic_state = %{atomics: %{first: %{}, second: %{}}}
              Map.put(acc, node_name, chemic_state)
            end)

          %{chemics: chemic_states, trace: []}
        end

        defp parse_dag(dag_block, nodes, edges) do
          # Initialize graph with all nodes having no dependencies
          graph =
            Enum.reduce(nodes || [], %{}, fn {node_name, _}, acc ->
              Map.put(acc, node_name, [])
            end)

          # Add edges
          edges =
            cond do
              is_nil(edges) -> []
              is_tuple(edges) -> [edges]
              is_list(edges) -> edges
              true -> []
            end

          Enum.reduce(edges, graph, fn {from, to}, acc ->
            Map.update(acc, to, [from], fn deps -> [from | deps] end)
          end)

          graph
        end

        defp extract_chemic_module(mod) when is_atom(mod), do: mod
        defp extract_chemic_module({:chemical, mod}), do: mod
        defp extract_chemic_module({:vertex, _, [mod]}), do: extract_chemic_module(mod)
        defp extract_chemic_module({:__aliases__, _, parts}), do: Module.concat(parts)

        defp extract_chemic_module_from_config({:__block__, _, expressions}) do
          extract_chemic_module_from_config(expressions)
        end

        defp extract_chemic_module_from_config(config) when is_list(config) do
          case Enum.find(config, fn {tag, _, _} -> tag == :vertex end) do
            {:vertex, _, [mod]} -> extract_chemic_module(mod)
            nil -> raise "No vertex found in node config"
          end
        end

        defp extract_chemic_module_from_config(config), do: extract_chemic_module(config)

        defp gather_node_inputs(data, []) do
          # For initial node, assume input is :value
          %{value: data.value}
        end

        defp gather_node_inputs(data, [dep | _]) do
          # For dependent node, use predecessor's output
          %{value: data[dep].result}
        end

        defp log_execution(node_name, inputs, outputs) do
          # Log execution for tracing
          # Update trace in state
          # For now, do nothing
        end

        defp extract_trace(state) do
          # Extract trace from state
          state.trace
        end

        # Extract outputs from final data
        defp extract_bionic_outputs(final_data, plan) do
          # Extract result from the last node in the plan
          last_node = List.last(plan)
          %{result: final_data[last_node][:result]}
        end

        defp verify_bionic_provenance(inputs, outputs) do
          # Verify provenance tracking through bionic execution
          AII.ProvenanceVerifier.verify_execution(inputs, outputs)
        end

        def run(inputs, opts \\ []) do
          metadata = __bionic_metadata__()

          # 1. Build execution plan
          {plan, dag} = build_execution_plan(metadata)

          # 2. Initialize state (use provided or default)
          initial_state = opts[:state] || initialize_bionic_state(metadata)

          # 3. Execute plan
          {final_state, final_data} =
            execute_plan(plan, dag, initial_state, inputs, opts, metadata)

          # 4. Extract outputs from final data
          outputs = extract_bionic_outputs(final_data, plan)

          # 5. Verify bionic provenance
          verify_bionic_provenance(inputs, outputs)

          # 6. Return outputs (and trace if requested)
          if opts[:trace] do
            {:ok, outputs, extract_trace(final_state)}
          else
            {:ok, outputs}
          end
        end

        def __bionic_metadata__ do
          %{
            name: @bionic_name,
            nodes: @nodes,
            edges: @edges,
            dag_block: @dag_block
          }
        end
      end
    end
  end

  defmacro dag(do: block) do
    quote do
      unquote(block)
    end
  end

  defmacro node(name, do: block) do
    quote do
      Module.put_attribute(__MODULE__, :current_node, unquote(name))
      Module.put_attribute(__MODULE__, :nodes, {unquote(name), unquote(Macro.escape(block))})
    end
  end

  defmacro edges(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :edges, unquote(Macro.escape(block)))
    end
  end

  defmacro inputs(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :inputs, unquote(Macro.escape(block)))
    end
  end

  defmacro outputs(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :outputs, unquote(Macro.escape(block)))
    end
  end

  defmacro vertex(module) do
    quote do
      unquote(module)
    end
  end

  defmacro edge(from) do
    quote do
      from_node =
        case unquote(from) do
          {:__aliases__, _, parts} ->
            List.last(parts) |> String.to_atom()

          atom when is_atom(atom) ->
            atom |> Atom.to_string() |> String.split(".") |> List.last() |> String.to_atom()

          _ ->
            raise "Invalid from"
        end

      current_node = Module.get_attribute(__MODULE__, :current_node)
      Module.put_attribute(__MODULE__, :edges, {from_node, current_node})
    end
  end

  defmacro chemical(name) do
    case name do
      {:__aliases__, _, parts} -> Module.concat(parts)
      atom when is_atom(atom) -> atom
      _ -> raise "Invalid module name"
    end
  end
end
