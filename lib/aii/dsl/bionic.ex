defmodule AII.DSL.Bionic do
  @moduledoc """
  DSL for orchestrating chemics into complete systems.
  Focus: End-to-end provenance verification.
  """

  defmacro defbionic(name, opts \\ [], do: block) do
    quote do
      defmodule Module.concat(:Bionic, unquote(name)) do
        @bionic_name unquote(name)
        @bionic_type unquote(opts[:type] || :orchestrator)
        @accelerator unquote(opts[:accelerator] || :cpu)

        use AII.Bionic

        unquote(block)

        @before_compile AII.Bionic
      end
    end
  end

  # Define inputs/outputs
  defmacro inputs(do: block) do
    quote do
      unquote(block)
    end
  end

  defmacro stream(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :input_streams, %{
        name: unquote(name),
        type: unquote(opts[:type])
      })
    end
  end

  defmacro context(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :context_data, %{
        name: unquote(name),
        type: unquote(opts[:type])
      })
    end
  end

  # Define DAG of chemics
  defmacro dag(do: block) do
    quote do
      unquote(block)
    end
  end

  defmacro node(name, do: block) do
    quote do
      Module.put_attribute(__MODULE__, :current_node_def, %{name: unquote(name)})
      unquote(block)
      node_def = Module.get_attribute(__MODULE__, :current_node_def)
      Module.put_attribute(__MODULE__, :dag_nodes, node_def)
    end
  end

  defmacro chemic(module) do
    quote do
      current = Module.get_attribute(__MODULE__, :current_node_def)

      Module.put_attribute(
        __MODULE__,
        :current_node_def,
        Map.put(current, :chemic, unquote(module))
      )
    end
  end

  # End-to-end provenance verification
  defmacro verify_end_to_end_provenance(do: block) do
    quote do
      def __end_to_end_verification__(var!(inputs), var!(outputs)) do
        unquote(block)
      end
    end
  end
end
