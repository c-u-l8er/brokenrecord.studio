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
      Module.register_attribute(__MODULE__, :accelerator_hint, [])

      # Default implementation
      def execute(atomic_state, inputs) do
        try do
          # 1. Run kernel
          outputs = kernel_function(atomic_state, inputs)

          # Ensure outputs is a map
          outputs = if is_map(outputs), do: outputs, else: %{result: outputs}

          # 2. Verify provenance
          :ok = AII.ProvenanceVerifier.verify_execution(inputs, outputs)

          {:ok, atomic_state, outputs}
        rescue
          error ->
            Logger.error("Error executing atomic #{__MODULE__}: #{inspect(error)}")
            reraise error, __STACKTRACE__
        end
      end
    end
  end
end
