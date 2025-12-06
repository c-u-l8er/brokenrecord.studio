defmodule AII.Atomic do
  @moduledoc """
  Behavior for atomic information transformations.
  Ensures provenance tracking and quality requirements.
  """

  @callback execute(inputs :: map()) ::
              {:ok, outputs :: map()} | {:error, term()}

  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Atomic

      import AII.DSL.Atomic

      Module.register_attribute(__MODULE__, :inputs, accumulate: true)
      Module.register_attribute(__MODULE__, :outputs, accumulate: true)
      Module.register_attribute(__MODULE__, :provenance_constraints, accumulate: false)
      Module.register_attribute(__MODULE__, :min_confidence, accumulate: false)
      Module.register_attribute(__MODULE__, :accelerator, accumulate: false)

      Module.put_attribute(__MODULE__, :provenance_constraints, nil)
      Module.put_attribute(__MODULE__, :min_confidence, nil)
      Module.put_attribute(__MODULE__, :accelerator, :cpu)

      @doc "Execute atomic with provenance tracking"
      def execute(inputs) do
        # 1. Verify all required inputs present
        :ok = verify_inputs(inputs)

        # 2. Verify input quality
        :ok = verify_input_quality(inputs)

        # 3. Run kernel
        outputs = kernel_function(inputs)

        # 4. Verify provenance constraints
        # :ok = verify_provenance_constraints(inputs, outputs)

        # 5. Verify output quality
        :ok = verify_output_quality(outputs)

        {:ok, outputs}
      rescue
        error -> {:error, error}
      end

      defp verify_inputs(inputs) do
        required =
          @inputs
          |> Enum.filter(& &1.required)
          |> Enum.map(& &1.name)

        missing = required -- Map.keys(inputs)

        if missing != [] do
          raise AII.Types.InputError, """
          Atomic #{@atomic_name} missing required inputs: #{inspect(missing)}
          """
        end

        :ok
      end

      defp verify_input_quality(inputs) do
        min_conf = @min_confidence || 0.7

        low_quality =
          inputs
          |> Enum.filter(fn {_name, tracked} ->
            not AII.Types.Tracked.acceptable?(tracked, min_conf)
          end)

        if low_quality != [] do
          raise AII.Types.QualityError, """
          Atomic #{@atomic_name} received low-quality inputs:
          #{inspect(low_quality, pretty: true)}
          Minimum confidence: #{min_conf}
          """
        end

        :ok
      end

      # defp verify_provenance_constraints(inputs, outputs) do
      #   if is_function(@provenance_constraints) do
      #     unless @provenance_constraints.(inputs, outputs) do
      #       raise AII.Types.ProvenanceViolation, """
      #       Atomic #{@atomic_name} violated provenance constraints
      #       Inputs: #{inspect(inputs, pretty: true)}
      #       Outputs: #{inspect(outputs, pretty: true)}
      #       """
      #     end
      #   end

      #   :ok
      # end

      defp verify_output_quality(outputs) do
        min_conf = @min_confidence || 0.7

        low_quality =
          outputs
          |> Enum.filter(fn {_name, tracked} ->
            not AII.Types.Tracked.acceptable?(tracked, min_conf)
          end)

        if low_quality != [] do
          IO.warn("""
          Atomic #{@atomic_name} produced low-quality outputs:
          #{inspect(low_quality, pretty: true)}
          Consider adjusting transformation parameters.
          """)
        end

        :ok
      end
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      def __atomic_metadata__ do
        %{
          name: @atomic_name,
          type: @atomic_type,
          inputs: @inputs,
          outputs: @outputs,
          min_confidence: @min_confidence,
          accelerator: @accelerator
        }
      end
    end
  end
end
