defmodule AII.DSL do
  @moduledoc "AII DSL with conservation types"

  defmacro __using__(_opts) do
    quote do
      import AII.DSL
      Module.register_attribute(__MODULE__, :agents, accumulate: true)
      Module.register_attribute(__MODULE__, :interactions, accumulate: true)
      Module.register_attribute(__MODULE__, :conserved_quantities, accumulate: true)

      def __agents__, do: @agents |> Enum.reverse()
      def __interactions__, do: @interactions |> Enum.reverse()
      def __conserved_quantities__, do: @conserved_quantities |> Enum.reverse()
    end
  end

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

  # New: conserves (conserved quantity for agent)
  defmacro conserves(quantities) do
    quote do
      Enum.each(unquote(quantities), fn quantity ->
        Module.put_attribute(__MODULE__, :conserved_quantities, quantity)
      end)
    end
  end

  # New: conserved_quantity (declare conserved quantities)
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

  # Modified: interaction with hardware hint
  defmacro interaction(name, opts \\ [], do: block) do
    with_target = Keyword.get(opts, :with)
    accelerator = Keyword.get(opts, :accelerator, :auto)

    quote do
      interaction = %{
        name: unquote(name),
        body: unquote(Macro.escape(block)),
        with: unquote(with_target),
        accelerator: unquote(accelerator)
      }
      Module.put_attribute(__MODULE__, :interactions, interaction)
    end
  end

  defmacro definteraction(name, opts \\ [], do: block) do
    accelerator = Keyword.get(opts, :accelerator, :auto)

    quote do
      interaction = %{
        name: unquote(name),
        body: unquote(Macro.escape(block)),
        accelerator: unquote(accelerator)
      }
      Module.put_attribute(__MODULE__, :interactions, interaction)
    end
  end

  # Agent definition macro
  defmacro defagent(name, do: block) do
    quote do
      agent_module = Module.concat(__MODULE__, unquote(name))

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

      # Register the agent in the parent module
      Module.put_attribute(__MODULE__, :agents, %{
        name: unquote(name),
        module: agent_module,
        fields: agent_module.__fields__(),
        interactions: agent_module.__interactions__(),
        conserved_quantities: agent_module.__conserved_quantities__()
      })
    end
  end

  # conserves macro for agents
  defmacro conserves(quantity1, quantity2, quantity3) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity1), unquote(quantity2), unquote(quantity3)])
    end
  end

  defmacro conserves(quantity1, quantity2, quantity3, quantity4) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity1), unquote(quantity2), unquote(quantity3), unquote(quantity4)])
    end
  end

  defmacro conserves(quantity1, quantity2, quantity3, quantity4, quantity5) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity1), unquote(quantity2), unquote(quantity3), unquote(quantity4), unquote(quantity5)])
    end
  end

  defmacro conserves(quantity1, quantity2) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity1), unquote(quantity2)])
    end
  end

  defmacro conserves(quantity) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, [unquote(quantity)])
    end
  end

  defmacro conserves(quantities) when is_list(quantities) do
    quote do
      Module.put_attribute(__MODULE__, :conserves, unquote(quantities))
    end
  end

end
