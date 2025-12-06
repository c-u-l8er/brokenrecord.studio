defmodule AII.Types do
  @moduledoc """
  Core types for AII: Provenance and Tracked for information, Conserved for physics.
  """

  # Provenance tracking for information transformations
  defmodule Provenance do
    @type t :: %__MODULE__{
            source_id: String.t(),
            source_type: source_type(),
            transformation_chain: [transformation()],
            created_at: DateTime.t(),
            confidence: float(),
            citations: [citation()],
            verified: boolean()
          }

    @type source_type ::
            :user_input
            | :database_query
            | :api_response
            | :file_upload
            | :sensor_reading
            | :computation
            | :llm_generation

    @type transformation :: %{
            atomic_name: atom(),
            operation: atom(),
            timestamp: DateTime.t(),
            confidence_before: float(),
            confidence_after: float(),
            parameters: map()
          }

    @type citation :: %{
            source: String.t(),
            url: String.t() | nil,
            authority_level: 1..10,
            verified_at: DateTime.t() | nil
          }

    defstruct [
      :source_id,
      :source_type,
      transformation_chain: [],
      created_at: DateTime.utc_now(),
      confidence: 1.0,
      citations: [],
      verified: false
    ]

    @doc "Create new provenance for original data"
    def new(source_id, source_type, opts \\ []) do
      %__MODULE__{
        source_id: source_id,
        source_type: source_type,
        created_at: Keyword.get(opts, :created_at, DateTime.utc_now()),
        confidence: Keyword.get(opts, :confidence, 1.0),
        citations: Keyword.get(opts, :citations, []),
        verified: Keyword.get(opts, :verified, false)
      }
    end

    @doc "Add transformation to provenance chain"
    def add_transformation(provenance, atomic_name, operation, new_confidence, params \\ %{}) do
      transformation = %{
        atomic_name: atomic_name,
        operation: operation,
        timestamp: DateTime.utc_now(),
        confidence_before: provenance.confidence,
        confidence_after: new_confidence,
        parameters: params
      }

      %{
        provenance
        | transformation_chain: [transformation | provenance.transformation_chain],
          confidence: new_confidence
      }
    end

    @doc "Merge multiple provenances (for aggregation operations)"
    def merge(provenances, operation) do
      # Take minimum confidence
      min_confidence =
        provenances
        |> Enum.map(& &1.confidence)
        |> Enum.min()

      # Combine all source IDs
      source_ids =
        provenances
        |> Enum.map(& &1.source_id)
        |> Enum.join(", ")

      # Combine citations
      all_citations =
        provenances
        |> Enum.flat_map(& &1.citations)
        |> Enum.uniq_by(& &1.source)

      %__MODULE__{
        source_id: "merged:#{source_ids}",
        source_type: :computation,
        transformation_chain: [
          %{
            atomic_name: :merge,
            operation: operation,
            timestamp: DateTime.utc_now(),
            confidence_before: min_confidence,
            # Slight degradation
            confidence_after: min_confidence * 0.9,
            parameters: %{merged_count: length(provenances)}
          }
        ],
        confidence: min_confidence * 0.9,
        citations: all_citations,
        verified: Enum.all?(provenances, & &1.verified)
      }
    end

    @doc "Check if provenance meets quality threshold"
    def acceptable?(provenance, min_confidence \\ 0.7) do
      provenance.confidence >= min_confidence
    end

    @doc "Get full transformation history"
    def history(provenance) do
      provenance.transformation_chain
      |> Enum.reverse()
      |> Enum.map(fn t ->
        """
        #{t.timestamp}: #{t.atomic_name}.#{t.operation}
          Confidence: #{t.confidence_before} → #{t.confidence_after}
          Params: #{inspect(t.parameters)}
        """
      end)
      |> Enum.join("\n")
    end
  end

  # Tracked value wrapper for information processing
  defmodule Tracked do
    @type t(inner) :: %__MODULE__{
            value: inner,
            provenance: Provenance.t(),
            metadata: map()
          }

    defstruct [:value, :provenance, metadata: %{}]

    @doc "Create tracked value with provenance"
    def new(value, source_id, source_type, opts \\ []) do
      %__MODULE__{
        value: value,
        provenance: Provenance.new(source_id, source_type, opts),
        metadata: Keyword.get(opts, :metadata, %{})
      }
    end

    @doc "Transform tracked value, updating provenance"
    def transform(tracked, atomic_name, operation, new_value, new_confidence, params \\ %{}) do
      %__MODULE__{
        value: new_value,
        provenance:
          Provenance.add_transformation(
            tracked.provenance,
            atomic_name,
            operation,
            new_confidence,
            params
          ),
        metadata: tracked.metadata
      }
    end

    @doc "Map over value while preserving provenance"
    def map(tracked, fun) do
      %{tracked | value: fun.(tracked.value)}
    end

    @doc "Check if tracked value is acceptable quality"
    def acceptable?(tracked, min_confidence \\ 0.7) do
      Provenance.acceptable?(tracked.provenance, min_confidence)
    end

    @doc "Aggregate multiple tracked values"
    def aggregate(tracked_values, aggregator_fn, operation) do
      # Apply aggregation to values
      aggregated_value = aggregator_fn.(Enum.map(tracked_values, & &1.value))

      # Merge provenances
      merged_provenance =
        Provenance.merge(
          Enum.map(tracked_values, & &1.provenance),
          operation
        )

      %__MODULE__{
        value: aggregated_value,
        provenance: merged_provenance,
        metadata: %{aggregated_from: length(tracked_values)}
      }
    end
  end

  # Conserved value wrapper for physics (unchanged)
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
          # Default for physics
          source_type: :computation,
          transformation_chain: [],
          created_at: DateTime.utc_now(),
          confidence: opts[:confidence] || 1.0,
          citations: [],
          verified: false
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
            source_type: :computation,
            transformation_chain: [],
            created_at: DateTime.utc_now(),
            confidence: 1.0,
            citations: [],
            verified: false
          }

      updated_provenance = %{
        current_provenance
        | transformation_chain: [transformation_type | current_provenance.transformation_chain]
      }

      %{conserved | provenance: updated_provenance}
    end
  end

  # Physics types (unchanged)
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

  # Geometric types (unchanged)
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

  # New exceptions for provenance
  defmodule InputError do
    defexception [:message]
  end

  defmodule QualityError do
    defexception [:message]
  end

  defmodule ProvenanceViolation do
    defexception [:message]
  end
end
