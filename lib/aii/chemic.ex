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

      Module.put_attribute(__MODULE__, :bonds, [])

      # Default pipeline provenance check - can be overridden
      def __pipeline_provenance_check__(_inputs, _outputs), do: true

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

        # 4. Verify pipeline provenance
        :ok = verify_pipeline_provenance(inputs, final_outputs)

        {:ok, final_outputs}
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
        atomic_def = Enum.find(@atomics, fn a -> a.name == atomic_name end)
        atomic_metadata = atomic_def.module.__atomic_metadata__()
        required_inputs = Enum.filter(atomic_metadata.inputs, & &1.required)

        # Find bonds that feed into this atomic
        input_bonds =
          @bonds
          |> Enum.filter(fn bond -> bond.to == atomic_name end)

        if input_bonds == [] do
          # For initial node, map chemic inputs to atomic inputs by name
          Enum.reduce(required_inputs, %{}, fn input_def, acc ->
            key = input_def.name
            Map.put(acc, key, inputs[key])
          end)
        else
          # Assume single dependency for now, map to :value
          bond = hd(input_bonds)
          %{value: state[bond.from].result}
        end
      end

      defp verify_pipeline_provenance(inputs, outputs) do
        unless __pipeline_provenance_check__(inputs, outputs) do
          raise AII.Types.ProvenanceViolation, """
          Chemic #{@chemic_name} violated pipeline provenance
          """
        end

        :ok
      end

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
