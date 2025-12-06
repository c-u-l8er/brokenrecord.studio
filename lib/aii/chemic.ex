defmodule AII.Chemic do
  @moduledoc """
  Behavior for chemic pipelines.
  Executes atomics in DAG order, propagating provenance.
  """

  @callback execute(inputs :: map()) ::
              {:ok, outputs :: map()} | {:error, term()}

  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Chemic

      import AII.DSL.Chemic

      Module.register_attribute(__MODULE__, :atomics, accumulate: true)
      Module.register_attribute(__MODULE__, :bonds, accumulate: false)
      Module.register_attribute(__MODULE__, :pipeline_provenance_check, accumulate: false)

      Module.put_attribute(__MODULE__, :bonds, [])
      Module.put_attribute(__MODULE__, :pipeline_provenance_check, nil)

      # Functions moved to __before_compile__
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      def execute(inputs) do
        # 1. Build execution DAG
        dag = build_dag(@atomics, @bonds)

        # 2. Topological sort
        execution_order = AII.Graph.topological_sort(dag)

        # 3. Execute atomics in order, propagating provenance
        {final_outputs, _state} =
          Enum.reduce(execution_order, {%{}, %{}}, fn atomic_name, {outputs, state} ->
            execute_atomic_node(atomic_name, outputs, state, inputs)
          end)

        # 4. Extract result from final outputs
        result_outputs =
          if execution_order == [],
            do: %{},
            else: %{result: final_outputs[:result]}

        # 5. Verify pipeline provenance (commented out)
        # :ok = verify_pipeline_provenance(inputs, result_outputs)

        {:ok, result_outputs}
      end

      defp execute_atomic_node(atomic_name, current_outputs, state, inputs) do
        atomic_def = Enum.find(@atomics, fn a -> a.name == atomic_name end)

        # Get inputs for this atomic from previous outputs or initial inputs
        atomic_inputs = gather_inputs_for(atomic_name, current_outputs, state, inputs)

        # Execute atomic
        {:ok, atomic_outputs} = atomic_def.module.execute(atomic_inputs)

        # Merge outputs
        merged_outputs = Map.merge(current_outputs, atomic_outputs)

        new_state = Map.put(state, atomic_name, atomic_outputs)

        {merged_outputs, new_state}
      end

      defp gather_inputs_for(atomic_name, outputs, state, inputs) do
        # Find bonds that feed into this atomic
        input_bonds =
          @bonds
          |> Enum.filter(fn bond -> bond.to == atomic_name end)

        if input_bonds == [] do
          # For initial node, assume input is :value
          %{value: inputs.value}
        else
          # Assume single dependency for now
          bond = hd(input_bonds)
          %{value: state[bond.from].result}
        end
      end

      # defp verify_pipeline_provenance(inputs, outputs) do
      #   if is_function(@pipeline_provenance_check) do
      #     unless @pipeline_provenance_check.(inputs, outputs) do
      #       raise AII.Types.ProvenanceViolation, """
      #       Chemic #{@chemic_name} violated pipeline provenance
      #       """
      #     end
      #   end

      #   :ok
      # end

      defp build_dag(atomics, bonds) do
        # Initialize graph with all atomic nodes having no dependencies
        atomic_names = Enum.map(atomics, & &1.name)
        graph = Map.new(atomic_names, &{&1, []})

        # Convert bonds to adjacency list: node => [dependencies]
        Enum.reduce(bonds, graph, fn bond, acc ->
          Map.update(acc, bond.to, [bond.from], fn deps ->
            [bond.from | deps]
          end)
        end)
      end

      def __chemic_metadata__ do
        %{
          name: @chemic_name,
          atomics: @atomics,
          bonds: @bonds
        }
      end
    end
  end
end
