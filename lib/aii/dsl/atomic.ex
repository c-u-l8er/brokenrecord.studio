defmodule AII.DSL.Atomic do
  @moduledoc """
  DSL for atomic information transformations.
  Focus: Provenance tracking, not conservation.
  """

  defmacro defatomic(name, opts \\ [], do: block) do
    quote do
      defmodule Module.concat(:Atomic, unquote(name)) do
        @atomic_name unquote(name)
        @atomic_type unquote(opts[:type] || :transform)
        @accelerator unquote(opts[:accelerator] || :cpu)

        use AII.Atomic

        unquote(block)

        @before_compile AII.Atomic
      end
    end
  end

  # Input declaration
  defmacro input(name, type, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :inputs, %{
        name: unquote(name),
        type: unquote(type),
        required: unquote(Keyword.get(opts, :required, true)),
        default: unquote(Keyword.get(opts, :default))
      })
    end
  end

  # Output declaration
  defmacro output(name, type, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :outputs, %{
        name: unquote(name),
        type: unquote(type),
        confidence_degradation: unquote(Keyword.get(opts, :confidence_degradation, 0.0))
      })
    end
  end

  # Provenance constraint
  defmacro tracks_provenance(do: block) do
    quote do
      def __provenance_check__(var!(inputs), var!(outputs)) do
        unquote(block)
      end
    end
  end

  # Quality constraint
  defmacro requires_quality(min_confidence) do
    quote do
      Module.put_attribute(__MODULE__, :min_confidence, unquote(min_confidence))
    end
  end

  # Main transformation kernel
  defmacro kernel(do: block) do
    quote do
      def kernel_function(var!(inputs)) do
        # Block has access to: input(:name), output(:name)
        # Returns: %{output_name: Tracked{value, provenance}}
        unquote(block)
      end
    end
  end

  # Accelerator hint
  defmacro accelerator(type) do
    quote do
      @accelerator unquote(type)
    end
  end
end
