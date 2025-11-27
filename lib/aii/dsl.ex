defmodule AII.DSL do
  @moduledoc "AII DSL with conservation types"

  # New: property (invariant field)
  defmacro property(name, type, opts \\ []) do
    invariant = Keyword.get(opts, :invariant, false)

    quote do
      field = %{
        name: unquote(name),
        type: unquote(type),
        invariant: unquote(invariant),
        kind: :property
      }
      Module.put_attribute(__MODULE__, :fields, field)
    end
  end

  # New: state (mutable field)
  defmacro state(name, type, _opts \\ []) do
    quote do
      field = %{
        name: unquote(name),
        type: unquote(type),
        invariant: false,
        kind: :state
      }
      Module.put_attribute(__MODULE__, :fields, field)
    end
  end

  # New: derives (computed field)
  defmacro derives(name, type, do: block) do
    quote do
      derived = %{
        name: unquote(name),
        type: unquote(type),
        computation: unquote(Macro.escape(block)),
        kind: :derived
      }
      Module.put_attribute(__MODULE__, :fields, derived)
    end
  end

  # Modified: interaction with hardware hint
  defmacro definteraction(name, opts \\ [], do: block) do
    accelerator = Keyword.get(opts, :accelerator, :auto)

    quote do
      interaction = %{
        name: unquote(name),
        body: unquote(Macro.escape(block)),
        accelerator: unquote(accelerator),
        conserved: []  # Filled by checker
      }
      Module.put_attribute(__MODULE__, :interactions, interaction)
    end
  end

  # New: conserved_quantity declaration
  defmacro conserved_quantity(name, opts \\ []) do
    quote do
      quantity = %{
        name: unquote(name),
        type: Keyword.get(unquote(opts), :type, :scalar),
        law: Keyword.get(unquote(opts), :law, :sum)
      }
      Module.put_attribute(__MODULE__, :conserved_quantities, quantity)
    end
  end

  # Agent definition macro
  defmacro defagent(name, do: block) do
    quote do
      defmodule unquote(name) do
        use AII.DSL
        Module.register_attribute(__MODULE__, :fields, accumulate: true)
        Module.register_attribute(__MODULE__, :interactions, accumulate: true)
        Module.register_attribute(__MODULE__, :conserved_quantities, accumulate: true)

        unquote(block)

        def __fields__, do: @fields |> Enum.reverse()
        def __interactions__, do: @interactions |> Enum.reverse()
        def __conserved_quantities__, do: @conserved_quantities |> Enum.reverse()
      end
    end
  end

  # conserves macro for agents
  defmacro conserves(quantity) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity)])
    end
  end

  defmacro conserves(quantity1, quantity2) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity1), unquote(quantity2)])
    end
  end

  defmacro __using__(_opts) do
    quote do
      import AII.DSL
    end
  end
end
