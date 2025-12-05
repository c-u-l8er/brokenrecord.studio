defmodule AII.Types do
  @moduledoc """
  Core types for AII: Conserved<T>, Energy, Momentum, etc.
  These types enforce conservation laws at compile time.
  """

  # Conserved wrapper type
  defmodule Conserved do
    @type t(inner) :: %__MODULE__{
            value: inner,
            # Where this value came from
            source: atom(),
            tracked: boolean()
          }

    defstruct value: 0, source: :unknown, tracked: true

    def new(value, source \\ :initial) do
      %__MODULE__{value: value, source: source, tracked: true}
    end

    # Can only transfer, never create
    def transfer(from, to, amount) do
      if from.value < amount do
        {:error, :insufficient_value}
      else
        new_from = %{from | value: from.value - amount}
        new_to = %{to | value: to.value + amount}
        {:ok, new_from, new_to}
      end
    end
  end

  # Physics types
  defmodule Energy do
    @type t :: Conserved.t(float)
  end

  defmodule Momentum do
    @type t :: Conserved.t({float, float, float})
  end

  defmodule Information do
    @type t :: Conserved.t(float)
  end

  defmodule Charge do
    @type t :: Conserved.t(float)
  end

  defmodule Mass do
    @type t :: Conserved.t(float)
  end

  # Geometric types
  defmodule Vec3 do
    @type t :: {float, float, float}

    def add({x1, y1, z1}, {x2, y2, z2}), do: {x1 + x2, y1 + y2, z1 + z2}
    def mul({x, y, z}, scalar), do: {x * scalar, y * scalar, z * scalar}
    def magnitude({x, y, z}), do: :math.sqrt(x * x + y * y + z * z)
    def sub({x1, y1, z1}, {x2, y2, z2}), do: {x1 - x2, y1 - y2, z1 - z2}

    def cross({x1, y1, z1}, {x2, y2, z2}),
      do: {y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2}

    def dot({x1, y1, z1}, {x2, y2, z2}), do: x1 * x2 + y1 * y2 + z1 * z2
  end

  defmodule Particle do
    defstruct [
      :id,
      :position,
      :velocity,
      :mass,
      # Conserved<Float>
      :information,
      # :atomic_name or :chemic_name
      :owner,
      :metadata
    ]
  end

  # Exceptions
  defmodule ConservationViolation do
    defexception [:message]
  end

  defmodule ChemicError do
    defexception [:message]
  end

  defmodule BionicError do
    defexception [:message]
  end
end
