defmodule AII.DSL.Chemic do
  defmacro defchemic(name, opts \\ [], do: block) do
    quote do
      defmodule Chemical.unquote(name) do
        use AII.Chemic

        @chemic_name unquote(name)

        Module.put_attribute(__MODULE__, :element_number, 1)
        Module.put_attribute(__MODULE__, :element_class, :basic)

        unquote(block)

        def execute(chemic_state, inputs) do
          try do
            metadata = __chemic_metadata__()
            # 1. Parse bonds into DAG
            dag = parse_bonds_to_dag(metadata.bonds, metadata.atomics)

            # 2. Topological sort
            execution_order = topological_sort(dag)

            # 3. Execute each node in order, accumulating data
            {updated_state, final_data} =
              Enum.reduce(execution_order, {chemic_state, inputs}, fn node, {state, data} ->
                {new_state, new_data} = execute_node(state, data, node, metadata.atomics, dag)
                {new_state, new_data}
              end)

            # 4. Extract outputs from final data
            outputs =
              if execution_order == [],
                do: %{},
                else: %{result: final_data[List.last(execution_order)].result}

            # 5. Verify chemic-level conservation
            verify_conservation(inputs, outputs)

            {:ok, updated_state, outputs}
          rescue
            error ->
              Logger.error("Error executing chemic #{__MODULE__}: #{inspect(error)}")
              reraise error, __STACKTRACE__
          end
        end

        def __chemic_metadata__ do
          %{
            name: @chemic_name,
            element_number: @element_number,
            element_class: @element_class,
            atomics: @atomics,
            sub_chemics: @sub_chemics,
            bonds: @bonds
          }
        end
      end
    end
  end

  defmacro composition(do: block) do
    quote do
      def __composition__ do
        unquote(block)
      end
    end
  end

  defmacro atomic(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :atomics, {unquote(name), unquote(opts)})
    end
  end

  defmacro chemic(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :sub_chemics, {unquote(name), unquote(opts)})
    end
  end

  defmacro bonds(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :bonds, unquote(Macro.escape(block)))
    end
  end
end
