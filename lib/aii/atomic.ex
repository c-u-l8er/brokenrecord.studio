defmodule AII.Atomic do
  @callback execute(atomic_state :: term(), inputs :: map()) ::
              {:ok, atomic_state :: term(), outputs :: map()} | {:error, term()}

  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Atomic

      import AII.DSL.Atomic

      require Logger

      Module.register_attribute(__MODULE__, :inputs, accumulate: true)
      Module.register_attribute(__MODULE__, :state_fields, accumulate: true)
      Module.register_attribute(__MODULE__, :conservation_laws, accumulate: true)
      Module.register_attribute(__MODULE__, :accelerator_hint, [])

      # Default implementation
      def execute(atomic_state, inputs) do
        try do
          # 1. Run kernel
          outputs = kernel_function(atomic_state, inputs)

          # Ensure outputs is a map
          outputs = if is_map(outputs), do: outputs, else: %{result: outputs}

          # 2. Verify conservation: outputs cannot create information
          :ok = AII.Conservation.verify(inputs, outputs)

          # 3. Verify output conservation laws if defined
          if @conservation_laws != [] do
            :ok = verify_output_conservation(inputs, outputs)
          end

          {:ok, atomic_state, outputs}
        rescue
          error ->
            Logger.error("Error executing atomic #{__MODULE__}: #{inspect(error)}")
            reraise error, __STACKTRACE__
        end
      end

      defp verify_output_conservation(inputs, outputs) do
        for {quantity, check_ast} <- @conservation_laws do
          # Evaluate the AST as a boolean expression
          # Bind inputs and outputs for evaluation
          bindings = [inputs: inputs, outputs: outputs]

          case Code.eval_quoted(check_ast, bindings) do
            {true, _} ->
              :ok

            {false, _} ->
              raise AII.Types.ConservationViolation, """
              Atomic #{inspect(@atomic_name)} violated conservation of #{quantity}
              Check: #{Macro.to_string(check_ast)}
              Inputs: #{inspect(inputs)}
              Outputs: #{inspect(outputs)}
              """

            _ ->
              raise AII.Types.ConservationViolation, """
              Atomic #{inspect(@atomic_name)} conservation check for #{quantity} did not return boolean
              Check: #{Macro.to_string(check_ast)}
              Inputs: #{inspect(inputs)}
              Outputs: #{inspect(outputs)}
              """
          end
        end

        :ok
      end
    end
  end
end
