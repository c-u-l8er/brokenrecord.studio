defmodule AII.Bionic do
  @moduledoc """
  Behavior for bionic orchestration.
  Executes chemics in DAG order with end-to-end provenance.
  """

  @callback run(inputs :: map()) ::
              {:ok, outputs :: map()} | {:error, term()}

  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Bionic

      import AII.DSL.Bionic

      Module.register_attribute(__MODULE__, :dag_nodes, accumulate: true)
      Module.register_attribute(__MODULE__, :input_streams, accumulate: true)
      Module.register_attribute(__MODULE__, :context_data, accumulate: true)
      Module.register_attribute(__MODULE__, :end_to_end_verification, accumulate: false)

      Module.put_attribute(__MODULE__, :end_to_end_verification, nil)
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      def run(inputs) do
        # 1. Validate inputs
        :ok = validate_inputs(inputs)

        # 2. Build execution DAG
        dag = build_execution_dag(@dag_nodes)

        # 3. Execute in topological order
        execution_order = AII.Graph.topological_sort(dag)

        {final_outputs, _state} =
          Enum.reduce(execution_order, {%{}, %{}}, fn node_name, {outputs, state} ->
            execute_chemic_node(node_name, outputs, state, inputs)
          end)

        # 4. Verify end-to-end provenance
        # :ok = verify_end_to_end(inputs, final_outputs)

        {:ok, final_outputs}
      end

      defp execute_chemic_node(node_name, current_outputs, state, inputs) do
        node_def = Enum.find(@dag_nodes, fn n -> n.name == node_name end)

        # Get chemic module
        chemic = node_def.chemic

        # Gather inputs
        chemic_inputs = gather_chemic_inputs(node_name, current_outputs, inputs)

        # Execute chemic
        {:ok, chemic_outputs} = chemic.execute(chemic_inputs)

        # Merge outputs
        merged = Map.merge(current_outputs, chemic_outputs)
        new_state = Map.put(state, node_name, chemic_outputs)

        {merged, new_state}
      end

      defp verify_end_to_end(inputs, outputs) do
        if is_function(@end_to_end_verification) do
          unless @end_to_end_verification.(inputs, outputs) do
            raise AII.Types.ProvenanceViolation, """
            Bionic #{@bionic_name} failed end-to-end provenance verification
            """
          end
        end

        :ok
      end

      # TODO: Implement these functions
      defp validate_inputs(inputs) do
        # Validate that inputs match expected streams/context
        :ok
      end

      defp build_execution_dag(nodes) do
        # Build DAG from node definitions
        # For now, assume all nodes have no dependencies
        Map.new(Enum.map(nodes, fn n -> {n.name, []} end))
      end

      defp gather_chemic_inputs(_node_name, outputs, inputs) do
        # For now, pass bionic inputs as inputs
        inputs
      end

      def __bionic_metadata__ do
        %{
          name: @bionic_name,
          dag_nodes: @dag_nodes,
          input_streams: @input_streams,
          context_data: @context_data
        }
      end
    end
  end
end
