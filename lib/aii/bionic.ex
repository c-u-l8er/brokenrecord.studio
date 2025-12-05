defmodule AII.Bionic do
  require Logger

  defmacro __using__(_opts) do
    quote do
      require Logger
      require AII.DSL.Bionic

      import AII.DSL.Bionic

      @tolerance 0.0001

      Module.register_attribute(__MODULE__, :nodes, accumulate: true)
      Module.register_attribute(__MODULE__, :edges, [])
      Module.register_attribute(__MODULE__, :dag_block, [])
    end
  end
end
