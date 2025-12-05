defmodule AII.Chemic do
  defmacro __using__(_opts) do
    quote do
      import AII.DSL.Chemic

      require Logger

      Module.register_attribute(__MODULE__, :atomics, accumulate: true)
      Module.register_attribute(__MODULE__, :sub_chemics, accumulate: true)
      Module.register_attribute(__MODULE__, :bonds, [])

      defp execute_node(state, data, node_name, atomics, dag) do
        # Get atomic module
        {^node_name, atomic_type} =
          Enum.find(atomics, fn {name, _} -> name == node_name end)

        atomic_module = atomic_type

        # Get atomic state
        atomic_state = Map.get(state.atomics, node_name)

        # Get inputs for this atomic based on dependencies
        dependencies = Map.get(dag, node_name, [])
        metadata = atomic_module.__atomic_metadata__()
        input_names = Enum.map(metadata.inputs, fn {name, _} -> name end)
        node_inputs = build_node_inputs(data, node_name, dependencies, input_names)

        # Execute atomic
        {:ok, updated_atomic, outputs} =
          atomic_module.execute(atomic_state, node_inputs)

        # Update state
        updated_state =
          put_in(state.atomics[node_name], updated_atomic)

        # Update data flow
        updated_data = Map.put(data, node_name, outputs)

        {updated_state, updated_data}
      end

      defp parse_bonds_to_dag(bonds_block, atomics) do
        # Get list of atomic node names
        atomic_names = Enum.map(atomics, fn {name, _} -> name end)

        # Initialize graph with all atomic nodes having no dependencies
        graph =
          Enum.reduce(atomic_names, %{}, fn name, acc ->
            Map.put(acc, name, [])
          end)

        # Normalize bonds block to list of tuples
        bonds =
          cond do
            is_nil(bonds_block) -> []
            is_list(bonds_block) -> bonds_block
            match?({:__block__, _, _}, bonds_block) -> elem(bonds_block, 2)
            true -> [bonds_block]
          end

        # Normalize -> syntax to {from, to}
        normalized_bonds =
          Enum.map(bonds, fn
            {:->, _, [from_ast, to_ast]} ->
              from = extract_ast(from_ast)
              to = extract_ast(to_ast)
              {from, to}

            other ->
              other
          end)

        # Add bonds only between atomic nodes
        Enum.reduce(normalized_bonds, graph, fn {from, to}, acc ->
          if from in atomic_names do
            Map.update(acc, to, [from], fn deps -> [from | deps] end)
          else
            acc
          end
        end)
      end

      defp topological_sort(dag) do
        AII.Graph.topological_sort(dag)
      rescue
        AII.Graph.CycleDetected ->
          raise AII.Types.ChemicError, "Cycle detected in bonds - DAG required"
      end

      defp build_adjacency_list(edges) do
        Enum.reduce(edges, %{}, fn {from, to}, acc ->
          Map.update(acc, to, [from], fn deps -> [from | deps] end)
        end)
      end

      defp extract_atom({atom, _, _}) when is_atom(atom), do: atom
      defp extract_atom(atom) when is_atom(atom), do: atom

      defp extract_ast(ast) when is_list(ast), do: extract_atom(hd(ast))
      defp extract_ast(ast), do: extract_atom(ast)

      defp build_node_inputs(data, node_name, dependencies, input_names) do
        if dependencies == [] do
          # Initial node: get from top-level input
          %{List.first(input_names) => data.value}
        else
          # Node with dependencies: map dependencies to input names
          Enum.zip(dependencies, input_names)
          |> Enum.into(%{}, fn {dep, name} -> {name, data[dep].result} end)
        end
      end

      defp verify_conservation(inputs, outputs) do
        input_info = AII.Conservation.total_information(inputs)
        output_info = AII.Conservation.total_information(outputs)
        diff = abs(input_info - output_info)
        tolerance = 0.0001

        if diff > tolerance do
          Logger.warning(
            "Chemic conservation violation: input #{input_info}, output #{output_info}, diff #{diff}"
          )
        end

        :ok
      end
    end
  end
end
