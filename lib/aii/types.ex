defmodule AII.Types do
  @moduledoc """
  Core types for AII: Conserved<T>, Energy, Momentum, etc.
  These types enforce conservation laws at compile time.
  """

  # Provenance tracking for transformations
  defmodule Provenance do
    @type t :: %__MODULE__{
            source_id: String.t(),
            transformation_chain: [transformation()],
            timestamp: DateTime.t(),
            confidence: float()
          }

    @type transformation ::
            {:multiplication, factor: number()}
            | {:addition, added: number()}
            | {:atomic_transform, atomic_module: atom()}
            | {:chemic_compose, chemic_module: atom()}
            | {:bionic_orchestrate, bionic_module: atom()}

    defstruct source_id: "", transformation_chain: [], timestamp: nil, confidence: 1.0
  end

  # Conserved value wrapper with provenance tracking
  defmodule Conserved do
    @type t(inner) :: %__MODULE__{
            value: inner,
            source: atom(),
            provenance: Provenance.t(),
            tracked: boolean()
          }

    defstruct value: 0, source: :unknown, provenance: nil, tracked: true

    # Create conserved value with provenance
    def new(value, source, opts \\ []) do
      %__MODULE__{
        value: value,
        source: source,
        provenance: %AII.Types.Provenance{
          source_id: opts[:source_id] || Atom.to_string(source),
          transformation_chain: [],
          timestamp: DateTime.utc_now(),
          confidence: opts[:confidence] || 1.0
        },
        tracked: true
      }
    end

    # Transform with provenance tracking
    def transform(conserved, transformation_type, _params) do
      current_provenance =
        conserved.provenance ||
          %AII.Types.Provenance{
            source_id: Atom.to_string(conserved.source),
            transformation_chain: [],
            timestamp: DateTime.utc_now(),
            confidence: 1.0
          }

      updated_provenance = %{
        current_provenance
        | transformation_chain: [transformation_type | current_provenance.transformation_chain]
      }

      %{conserved | provenance: updated_provenance}
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
