defmodule AII.DSL.Chemic do
  @moduledoc """
  DSL for composing atomics into pipelines.
  Focus: Provenance flows through transformations.
  """

  defmacro defchemic(name, opts \\ [], do: block) do
    quote do
      defmodule Module.concat(:Chemic, unquote(name)) do
        @chemic_name unquote(name)

        use AII.Chemic

        unquote(block)

        @before_compile AII.Chemic
      end
    end
  end

  # Declare atomics in chemic
  defmacro atomic(name, module) do
    quote do
      Module.put_attribute(__MODULE__, :atomics, %{
        name: unquote(name),
        module: unquote(module)
      })
    end
  end

  # Declare bonds (data flow)
  defmacro bonds(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :temp_bonds, [])
      unquote(block)
      bonds = Module.get_attribute(__MODULE__, :temp_bonds)
      Module.put_attribute(__MODULE__, :bonds, bonds)
    end
  end

  # Bond syntax: bond(from, to)
  defmacro bond(from, to) do
    quote do
      current = Module.get_attribute(__MODULE__, :temp_bonds) || []

      bond = %{
        from: unquote(from),
        to: unquote(to)
      }

      Module.put_attribute(__MODULE__, :temp_bonds, [bond | current])
    end
  end

  # Tracks provenance through entire pipeline
  defmacro tracks_pipeline_provenance(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :pipeline_provenance_check, fn inputs, outputs ->
        unquote(block)
      end)
    end
  end
end
