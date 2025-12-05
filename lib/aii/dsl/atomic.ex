defmodule AII.DSL.Atomic do
  defmacro defatomic(name, opts \\ [], do: block) do
    quote do
      defmodule Atomical.unquote(name) do
        use AII.Atomic

        @atomic_name __MODULE__ |> Atom.to_string() |> String.replace("Elixir.Atomical.", "")
        @accelerator nil

        Module.put_attribute(__MODULE__, :atomic_type, unquote(opts[:type] || :basic))

        unquote(block)

        def __atomic_metadata__ do
          %{
            name: @atomic_name,
            type: @atomic_type,
            inputs: @inputs,
            state_fields: @state_fields,
            conservation_laws: @conservation_laws,
            accelerator: @accelerator
          }
        end
      end
    end
  end

  defmacro atomical(name) do
    quote do
      @atomic_name unquote(name)
    end
  end

  defmacro kernel(do: block) do
    quote do
      def kernel_function(atomic_state, var!(inputs)) do
        unquote(block)
      end
    end
  end

  defmacro input(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :inputs, {unquote(name), unquote(opts)})
    end
  end

  defmacro state(name, _opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :state_fields, unquote(name))
    end
  end

  defmacro conserves(quantity, do: block) do
    quote do
      Module.put_attribute(
        __MODULE__,
        :conservation_laws,
        {unquote(quantity), unquote(Macro.escape(block))}
      )
    end
  end

  defmacro transform(do: block) do
    quote do
      def transform_function(atomic_state, inputs) do
        unquote(block)
      end
    end
  end

  defmacro accelerator(type) do
    quote do
      @accelerator unquote(type)
    end
  end
end
