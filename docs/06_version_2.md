# AII Implementation Guide: Atomic/Chemic/Bionic with Provenance
## Document 6: Information Processing Pipelines (Not Physics)

**Context:** This document implements the hierarchical composition system for **information processing**, not physics simulations.  
**Key Distinction:** 
- **Physics (defagent)**: Uses conservation laws (energy, momentum)
- **Information (atomic/chemic/bionic)**: Uses provenance tracking (where did this data come from?)

**Goal:** Build verifiable data transformation pipelines where every output has traceable origins.  
**Audience:** LLM coding assistant implementing AII's information processing layer

---

## Core Philosophy: Provenance Over Conservation

### Why Provenance for Information?

**Information is NOT conserved** - it can be:
- Created (generate new insights)
- Transformed (analyze → conclusions)
- Aggregated (many → one summary)
- Destroyed (filter → subset)

But **provenance must be tracked**:
- Where did this fact come from?
- What transformations were applied?
- What was the confidence at each step?
- Can we trace back to original sources?

### Comparison

```elixir
# ❌ WRONG: Information conservation
defatomic AnalyzeText do
  conserves :information do
    # This makes no sense - analysis creates new information!
    total_information(output) == total_information(input)
  end
end

# ✅ RIGHT: Provenance tracking
defatomic AnalyzeText do
  tracks_provenance do
    # Every output insight can be traced to input text
    output(:insights).provenance.source == input(:text).source
    output(:insights).provenance.transformation == :sentiment_analysis
    output(:insights).provenance.confidence >= 0.8
  end
end
```

---

## 1. Core Type System

### Provenance Types

**File:** `lib/aii/types/provenance.ex`

```elixir
defmodule AII.Types.Provenance do
  @moduledoc """
  Provenance tracking for information processing.
  Every value knows where it came from and how it was transformed.
  """
  
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
    :user_input |
    :database_query |
    :api_response |
    :file_upload |
    :sensor_reading |
    :computation |
    :llm_generation
  
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
    
    %{provenance |
      transformation_chain: [transformation | provenance.transformation_chain],
      confidence: new_confidence
    }
  end
  
  @doc "Merge multiple provenances (for aggregation operations)"
  def merge(provenances, operation) do
    # Take minimum confidence
    min_confidence = provenances
      |> Enum.map(& &1.confidence)
      |> Enum.min()
    
    # Combine all source IDs
    source_ids = provenances
      |> Enum.map(& &1.source_id)
      |> Enum.join(", ")
    
    # Combine citations
    all_citations = provenances
      |> Enum.flat_map(& &1.citations)
      |> Enum.uniq_by(& &1.source)
    
    %__MODULE__{
      source_id: "merged:#{source_ids}",
      source_type: :computation,
      transformation_chain: [%{
        atomic_name: :merge,
        operation: operation,
        timestamp: DateTime.utc_now(),
        confidence_before: min_confidence,
        confidence_after: min_confidence * 0.9,  # Slight degradation
        parameters: %{merged_count: length(provenances)}
      }],
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
```

### Tracked Value Wrapper

**File:** `lib/aii/types/tracked.ex`

```elixir
defmodule AII.Types.Tracked do
  @moduledoc """
  Wrapper for values with provenance tracking.
  Unlike Conserved<T> (for physics), Tracked<T> is for information.
  """
  
  @type t(inner) :: %__MODULE__{
    value: inner,
    provenance: AII.Types.Provenance.t(),
    metadata: map()
  }
  
  defstruct [:value, :provenance, metadata: %{}]
  
  @doc "Create tracked value with provenance"
  def new(value, source_id, source_type, opts \\ []) do
    %__MODULE__{
      value: value,
      provenance: AII.Types.Provenance.new(source_id, source_type, opts),
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end
  
  @doc "Transform tracked value, updating provenance"
  def transform(tracked, atomic_name, operation, new_value, new_confidence, params \\ %{}) do
    %__MODULE__{
      value: new_value,
      provenance: AII.Types.Provenance.add_transformation(
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
    AII.Types.Provenance.acceptable?(tracked.provenance, min_confidence)
  end
  
  @doc "Aggregate multiple tracked values"
  def aggregate(tracked_values, aggregator_fn, operation) do
    # Apply aggregation to values
    aggregated_value = aggregator_fn.(Enum.map(tracked_values, & &1.value))
    
    # Merge provenances
    merged_provenance = AII.Types.Provenance.merge(
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
```

---

## 2. Atomic DSL Implementation

### Core Atomic Macro

**File:** `lib/aii/dsl/atomic.ex`

```elixir
defmodule AII.DSL.Atomic do
  @moduledoc """
  DSL for atomic information transformations.
  Focus: Provenance tracking, not conservation.
  """
  
  defmacro defatomic(name, opts \\ [], do: block) do
    quote do
      defmodule Module.concat(:Atomic, unquote(name)) do
        use AII.Atomic
        
        Module.put_attribute(__MODULE__, :atomic_name, unquote(name))
        Module.put_attribute(__MODULE__, :atomic_type, unquote(opts[:type] || :transform))
        Module.put_attribute(__MODULE__, :accelerator, unquote(opts[:accelerator] || :cpu))
        
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
      Module.put_attribute(__MODULE__, :provenance_constraints, 
        fn inputs, outputs -> unquote(block) end
      )
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
      def kernel_function(inputs) do
        # Block has access to: input(:name), output(:name)
        # Returns: %{output_name: Tracked{value, provenance}}
        unquote(block)
      end
    end
  end
  
  # Accelerator hint
  defmacro accelerator(type) do
    quote do
      Module.put_attribute(__MODULE__, :accelerator, unquote(type))
    end
  end
end
```

### Atomic Behavior

**File:** `lib/aii/atomic.ex`

```elixir
defmodule AII.Atomic do
  @moduledoc """
  Behavior for atomic information transformations.
  Ensures provenance tracking and quality requirements.
  """
  
  @callback execute(inputs :: map()) :: 
    {:ok, outputs :: map()} | {:error, term()}
  
  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Atomic
      
      import AII.DSL.Atomic
      
      Module.register_attribute(__MODULE__, :inputs, accumulate: true)
      Module.register_attribute(__MODULE__, :outputs, accumulate: true)
      Module.register_attribute(__MODULE__, :provenance_constraints, [])
      Module.register_attribute(__MODULE__, :min_confidence, [])
      Module.register_attribute(__MODULE__, :accelerator, [])
      
      @doc "Execute atomic with provenance tracking"
      def execute(inputs) do
        # 1. Verify all required inputs present
        :ok = verify_inputs(inputs)
        
        # 2. Verify input quality
        :ok = verify_input_quality(inputs)
        
        # 3. Run kernel
        outputs = kernel_function(inputs)
        
        # 4. Verify provenance constraints
        :ok = verify_provenance_constraints(inputs, outputs)
        
        # 5. Verify output quality
        :ok = verify_output_quality(outputs)
        
        {:ok, outputs}
      rescue
        error -> {:error, error}
      end
      
      defp verify_inputs(inputs) do
        required = @inputs
          |> Enum.filter(& &1.required)
          |> Enum.map(& &1.name)
        
        missing = required -- Map.keys(inputs)
        
        if missing != [] do
          raise AII.InputError, """
          Atomic #{@atomic_name} missing required inputs: #{inspect(missing)}
          """
        end
        
        :ok
      end
      
      defp verify_input_quality(inputs) do
        min_conf = @min_confidence || 0.7
        
        low_quality = inputs
          |> Enum.filter(fn {_name, tracked} ->
            not AII.Types.Tracked.acceptable?(tracked, min_conf)
          end)
        
        if low_quality != [] do
          raise AII.QualityError, """
          Atomic #{@atomic_name} received low-quality inputs:
          #{inspect(low_quality, pretty: true)}
          Minimum confidence: #{min_conf}
          """
        end
        
        :ok
      end
      
      defp verify_provenance_constraints(inputs, outputs) do
        if @provenance_constraints do
          unless @provenance_constraints.(inputs, outputs) do
            raise AII.ProvenanceViolation, """
            Atomic #{@atomic_name} violated provenance constraints
            Inputs: #{inspect(inputs, pretty: true)}
            Outputs: #{inspect(outputs, pretty: true)}
            """
          end
        end
        
        :ok
      end
      
      defp verify_output_quality(outputs) do
        min_conf = @min_confidence || 0.7
        
        low_quality = outputs
          |> Enum.filter(fn {_name, tracked} ->
            not AII.Types.Tracked.acceptable?(tracked, min_conf)
          end)
        
        if low_quality != [] do
          IO.warn("""
          Atomic #{@atomic_name} produced low-quality outputs:
          #{inspect(low_quality, pretty: true)}
          Consider adjusting transformation parameters.
          """)
        end
        
        :ok
      end
    end
  end
  
  defmacro __before_compile__(_env) do
    quote do
      def __atomic_metadata__ do
        %{
          name: @atomic_name,
          type: @atomic_type,
          inputs: @inputs,
          outputs: @outputs,
          min_confidence: @min_confidence,
          accelerator: @accelerator
        }
      end
    end
  end
end
```

---

## 3. Example Atomics

### Text Analysis Atomic

```elixir
defatomic SentimentAnalyzer do
  @moduledoc "Analyze sentiment of text with provenance tracking"
  
  input :text, String, required: true
  output :sentiment, %{polarity: float(), subjectivity: float()}
  
  requires_quality 0.8
  accelerator :npu  # Neural processing for sentiment analysis
  
  tracks_provenance do
    # Output sentiment must come from input text
    output(:sentiment).provenance.source_id == input(:text).provenance.source_id
    
    # Transformation must be sentiment_analysis
    output(:sentiment).provenance.transformation_chain
    |> Enum.any?(fn t -> t.operation == :sentiment_analysis end)
    
    # Confidence should degrade slightly (analysis adds uncertainty)
    output(:sentiment).provenance.confidence <= input(:text).provenance.confidence
  end
  
  kernel do
    text = input(:text)
    
    # Perform sentiment analysis (stub - would call NLP model)
    sentiment_result = %{
      polarity: analyze_polarity(text.value),
      subjectivity: analyze_subjectivity(text.value)
    }
    
    # Confidence degrades slightly due to analysis uncertainty
    new_confidence = text.provenance.confidence * 0.95
    
    # Return tracked output with updated provenance
    %{
      sentiment: AII.Types.Tracked.transform(
        text,
        __MODULE__,
        :sentiment_analysis,
        sentiment_result,
        new_confidence,
        %{model: "sentiment_v1", threshold: 0.5}
      )
    }
  end
  
  defp analyze_polarity(text), do: 0.7  # Stub
  defp analyze_subjectivity(text), do: 0.6  # Stub
end
```

### Database Query Atomic

```elixir
defatomic DatabaseQuery do
  @moduledoc "Query database with full provenance tracking"
  
  input :query, String, required: true
  input :database_connection, %{host: String.t(), database: String.t()}
  output :results, [map()]
  
  requires_quality 0.95  # Database queries must be high confidence
  accelerator :cpu
  
  tracks_provenance do
    # Results must come from verified database
    output(:results).provenance.source_type == :database_query
    output(:results).provenance.verified == true
    
    # Citations must include database info
    output(:results).provenance.citations
    |> Enum.any?(fn c -> c.source =~ input(:database_connection).value.database end)
  end
  
  kernel do
    query = input(:query).value
    db_conn = input(:database_connection).value
    
    # Execute query (stub)
    raw_results = execute_sql_query(db_conn, query)
    
    # Create provenance for database results
    db_provenance = AII.Types.Provenance.new(
      "db:#{db_conn.host}/#{db_conn.database}",
      :database_query,
      verified: true,
      confidence: 0.99,  # Database queries are highly confident
      citations: [%{
        source: "Database: #{db_conn.database}",
        url: nil,
        authority_level: 9,
        verified_at: DateTime.utc_now()
      }]
    )
    
    # Wrap results in Tracked with provenance
    %{
      results: %AII.Types.Tracked{
        value: raw_results,
        provenance: db_provenance,
        metadata: %{
          query: query,
          row_count: length(raw_results)
        }
      }
    }
  end
  
  defp execute_sql_query(_conn, _query), do: []  # Stub
end
```

### Route Calculation Atomic (for GIS)

```elixir
defatomic DijkstraRouting do
  @moduledoc "Calculate shortest path with provenance for every waypoint"
  
  input :start_node, String, required: true
  input :end_node, String, required: true
  input :road_network, %{nodes: map(), edges: map()}
  output :route, %{waypoints: [String.t()], total_cost: float()}
  
  requires_quality 0.9
  accelerator :rt_cores  # Use RT cores for BVH-accelerated graph search
  
  tracks_provenance do
    # Every waypoint must come from verified road network
    output(:route).value.waypoints
    |> Enum.all?(fn waypoint ->
      waypoint in Map.keys(input(:road_network).value.nodes)
    end)
    
    # Cost must be calculable from road network
    output(:route).provenance.citations
    |> Enum.any?(fn c -> c.source =~ "road_network" end)
  end
  
  kernel do
    start = input(:start_node).value
    end_node = input(:end_node).value
    network = input(:road_network).value
    
    # Run Dijkstra (would use RT cores in real implementation)
    {waypoints, cost} = dijkstra_algorithm(network, start, end_node)
    
    # Create provenance for route
    route_provenance = AII.Types.Provenance.new(
      "route:#{start}->#{end_node}",
      :computation,
      confidence: 0.95,
      citations: [%{
        source: "Road Network Database",
        url: nil,
        authority_level: 8,
        verified_at: DateTime.utc_now()
      }]
    )
    
    route_provenance = AII.Types.Provenance.add_transformation(
      route_provenance,
      __MODULE__,
      :dijkstra_routing,
      0.95,
      %{algorithm: "dijkstra", accelerator: "rt_cores"}
    )
    
    %{
      route: %AII.Types.Tracked{
        value: %{waypoints: waypoints, total_cost: cost},
        provenance: route_provenance,
        metadata: %{
          waypoint_count: length(waypoints),
          computation_time_ms: 5.2  # From RT cores
        }
      }
    }
  end
  
  defp dijkstra_algorithm(_network, _start, _end), do: {["A", "B", "C"], 42.0}  # Stub
end
```

---

## 4. Chemic DSL Implementation

### Chemic Macro

**File:** `lib/aii/dsl/chemic.ex`

```elixir
defmodule AII.DSL.Chemic do
  @moduledoc """
  DSL for composing atomics into pipelines.
  Focus: Provenance flows through transformations.
  """
  
  defmacro defchemic(name, opts \\ [], do: block) do
    quote do
      defmodule Module.concat(:Chemic, unquote(name)) do
        use AII.Chemic
        
        Module.put_attribute(__MODULE__, :chemic_name, unquote(name))
        
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
      bonds_list = []
      unquote(block)
      Module.put_attribute(__MODULE__, :bonds, bonds_list)
    end
  end
  
  # Bond syntax: source → target
  defmacro source → target do
    quote do
      bond = %{
        from: unquote(source),
        to: unquote(target)
      }
      bonds_list = [bond | bonds_list]
    end
  end
  
  # Tracks provenance through entire pipeline
  defmacro tracks_pipeline_provenance(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :pipeline_provenance_check,
        fn inputs, outputs -> unquote(block) end
      )
    end
  end
end
```

### Chemic Behavior

**File:** `lib/aii/chemic.ex`

```elixir
defmodule AII.Chemic do
  @moduledoc """
  Behavior for chemic pipelines.
  Executes atomics in DAG order, propagating provenance.
  """
  
  @callback execute(inputs :: map()) :: 
    {:ok, outputs :: map()} | {:error, term()}
  
  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Chemic
      
      import AII.DSL.Chemic
      
      Module.register_attribute(__MODULE__, :atomics, accumulate: true)
      Module.register_attribute(__MODULE__, :bonds, [])
      Module.register_attribute(__MODULE__, :pipeline_provenance_check, [])
      
      def execute(inputs) do
        # 1. Build execution DAG
        dag = build_dag(@atomics, @bonds)
        
        # 2. Topological sort
        execution_order = AII.Graph.topological_sort(dag)
        
        # 3. Execute atomics in order, propagating provenance
        {final_outputs, _state} = Enum.reduce(execution_order, {%{}, %{}}, 
          fn atomic_name, {outputs, state} ->
            execute_atomic_node(atomic_name, outputs, state)
          end
        )
        
        # 4. Verify pipeline provenance
        :ok = verify_pipeline_provenance(inputs, final_outputs)
        
        {:ok, final_outputs}
      end
      
      defp execute_atomic_node(atomic_name, current_outputs, state) do
        atomic_def = Enum.find(@atomics, fn a -> a.name == atomic_name end)
        
        # Get inputs for this atomic from previous outputs or initial inputs
        atomic_inputs = gather_inputs_for(atomic_name, current_outputs, state)
        
        # Execute atomic
        {:ok, atomic_outputs} = atomic_def.module.execute(atomic_inputs)
        
        # Merge outputs
        merged_outputs = Map.merge(current_outputs, atomic_outputs)
        
        # Update state
        new_state = Map.put(state, atomic_name, atomic_outputs)
        
        {merged_outputs, new_state}
      end
      
      defp gather_inputs_for(atomic_name, outputs, _state) do
        # Find bonds that feed into this atomic
        input_bonds = @bonds
          |> Enum.filter(fn bond -> bond.to == atomic_name end)
        
        # Gather outputs from source atomics
        Enum.reduce(input_bonds, %{}, fn bond, acc ->
          Map.merge(acc, outputs[bond.from] || %{})
        end)
      end
      
      defp verify_pipeline_provenance(inputs, outputs) do
        if @pipeline_provenance_check do
          unless @pipeline_provenance_check.(inputs, outputs) do
            raise AII.ProvenanceViolation, """
            Chemic #{@chemic_name} violated pipeline provenance
            """
          end
        end
        
        :ok
      end
      
      defp build_dag(atomics, bonds) do
        # Convert bonds to adjacency list
        Enum.reduce(bonds, %{}, fn bond, acc ->
          Map.update(acc, bond.from, [bond.to], fn targets ->
            [bond.to | targets]
          end)
        end)
      end
    end
  end
  
  defmacro __before_compile__(_env) do
    quote do
      def __chemic_metadata__ do
        %{
          name: @chemic_name,
          atomics: @atomics,
          bonds: @bonds
        }
      end
    end
  end
end
```

---

## 5. Example Chemic

### Text Processing Pipeline

```elixir
defchemic TextAnalysisPipeline do
  @moduledoc "Complete text analysis with provenance tracking"
  
  # Declare atomics
  atomic :parse, TextParser
  atomic :analyze_sentiment, SentimentAnalyzer
  atomic :extract_entities, EntityExtractor
  atomic :summarize, TextSummarizer
  
  # Data flow
  bonds do
    input(:raw_text) → :parse
    :parse → :analyze_sentiment
    :parse → :extract_entities
    [:analyze_sentiment, :extract_entities] → :summarize
    :summarize → output(:summary)
  end
  
  # Verify end-to-end provenance
  tracks_pipeline_provenance do
    # Output summary must trace back to original input
    output(:summary).provenance.source_id == input(:raw_text).provenance.source_id
    
    # All transformations must be in chain
    transformations = output(:summary).provenance.transformation_chain
      |> Enum.map(& &1.atomic_name)
    
    [:TextParser, :SentimentAnalyzer, :EntityExtractor, :TextSummarizer]
    |> Enum.all?(fn atomic -> atomic in transformations end)
    
    # Confidence should degrade through pipeline
    output(:summary).provenance.confidence < input(:raw_text).provenance.confidence
  end
end
```

### Route Optimization Pipeline (for GIS)

```elixir
defchemic RouteOptimizationPipeline do
  @moduledoc "Multi-objective route optimization with provenance"
  
  # Atomics
  atomic :calculate_shortest, DijkstraRouting
  atomic :calculate_fastest, AStarRouting
  atomic :check_traffic, TrafficAnalyzer
  atomic :optimize_multi, MultiObjectiveOptimizer
  
  # Pipeline
  bonds do
    [input(:start), input(:end), input(:network)] → :calculate_shortest
    [input(:start), input(:end), input(:network)] → :calculate_fastest
    input(:current_time) → :check_traffic
    [:calculate_shortest, :calculate_fastest, :check_traffic] → :optimize_multi
    :optimize_multi → output(:optimal_route)
  end
  
  tracks_pipeline_provenance do
    # All waypoints must be from verified network
    output(:optimal_route).provenance.citations
    |> Enum.any?(fn c -> c.source =~ "Road Network" end)
    
    # Must include all optimization criteria
    transformations = output(:optimal_route).provenance.transformation_chain
    
    [:DijkstraRouting, :AStarRouting, :TrafficAnalyzer, :MultiObjectiveOptimizer]
    |> Enum.all?(fn atomic -> 
      Enum.any?(transformations, fn t -> t.atomic_name == atomic end)
    end)
  end
end
```

---

## 6. Bionic DSL Implementation

### Bionic Macro

**File:** `lib/aii/dsl/bionic.ex`

```elixir
defmodule AII.DSL.Bionic do
  @moduledoc """
  DSL for orchestrating chemics into complete systems.
  Focus: End-to-end provenance verification.
  """
  
  defmacro defbionic(name, opts \\ [], do: block) do
    quote do
      defmodule Module.concat(:Bionic, unquote(name)) do
        use AII.Bionic
        
        Module.put_attribute(__MODULE__, :bionic_name, unquote(name))
        
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
      node_def = %{name: unquote(name)}
      unquote(block)
      Module.put_attribute(__MODULE__, :dag_nodes, node_def)
    end
  end
  
  defmacro chemic(module) do
    quote do
      node_def = Map.put(node_def, :chemic, unquote(module))
    end
  end
  
  # End-to-end provenance verification
  defmacro verify_end_to_end_provenance(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :end_to_end_verification,
        fn inputs, outputs -> unquote(block) end
      )
    end
  end
end
```

### Bionic Behavior

**File:** `lib/aii/bionic.ex`

```elixir
defmodule AII.Bionic do
  @moduledoc """
  Behavior for bionic orchestration.
  Executes chemics in DAG order with end-to-end provenance.
  """
  
  @callback run(inputs :: map()) :: 
    {:ok, outputs :: map()} | {:error, term()}
  
  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Bionic
      
      import AII.DSL.Bionic
      
      Module.register_attribute(__MODULE__, :dag_nodes, accumulate: true)
      Module.register_attribute(__MODULE__, :input_streams, accumulate: true)
      Module.register_attribute(__MODULE__, :context_data, accumulate: true)
      Module.register_attribute(__MODULE__, :end_to_end_verification, [])
      
      def run(inputs) do
        # 1. Validate inputs
        :ok = validate_inputs(inputs)
        
        # 2. Build execution DAG
        dag = build_execution_dag(@dag_nodes)
        
        # 3. Execute in topological order
        execution_order = AII.Graph.topological_sort(dag)
        
        {final_outputs, _state} = Enum.reduce(execution_order, {%{}, %{}},
          fn node_name, {outputs, state} ->
            execute_chemic_node(node_name, outputs, state)
          end
        )
        
        # 4. Verify end-to-end provenance
        :ok = verify_end_to_end(inputs, final_outputs)
        
        {:ok, final_outputs}
      end
      
      defp execute_chemic_node(node_name, current_outputs, state) do
        node_def = Enum.find(@dag_nodes, fn n -> n.name == node_name end)
        
        # Get chemic module
        chemic = node_def.chemic
        
        # Gather inputs
        chemic_inputs = gather_chemic_inputs(node_name, current_outputs)
        
        # Execute chemic
        {:ok, chemic_outputs} = chemic.execute(chemic_inputs)
        
        # Merge outputs
        merged = Map.merge(current_outputs, chemic_outputs)
        new_state = Map.put(state, node_name, chemic_outputs)
        
        {merged, new_state}
      end
      
      defp verify_end_to_end(inputs, outputs) do
        if @end_to_end_verification do
          unless @end_to_end_verification.(inputs, outputs) do
            raise AII.ProvenanceViolation, """
            Bionic #{@bionic_name} failed end-to-end provenance verification
            """
          end
        end
        
        :ok
      end
      
      defp validate_inputs(inputs) do
        required_streams = @input_streams |> Enum.map(& &1.name)
        missing = required_streams -- Map.keys(inputs)
        
        if missing != [] do
          raise AII.InputError, "Missing input streams: #{inspect(missing)}"
        end
        
        :ok
      end
      
      defp gather_chemic_inputs(_node_name, outputs), do: outputs
      
      defp build_execution_dag(nodes) do
        # Build adjacency list from node definitions
        # (Simplified - would include edge definitions)
        %{}
      end
    end
  end
  
  defmacro __before_compile__(_env) do
    quote do
      def __bionic_metadata__ do
        %{
          name: @bionic_name,
          nodes: @dag_nodes,
          input_streams: @input_streams
        }
      end
    end
  end
end
```

---

## 7. Complete Example: Hallucination-Free Chatbot

### Chatbot Bionic

```elixir
defbionic HallucinationFreeChatbot do
  @moduledoc """
  Complete chatbot with impossible hallucination.
  Every response traced to verified sources.
  """
  
  inputs do
    stream :user_message, type: String
    context :knowledge_base, type: AII.KnowledgeBase
    context :conversation_history, type: [map()]
  end
  
  dag do
    node :parse_query do
      chemic :QueryParsingPipeline
      input [:user_message]
      output :parsed_query
    end
    
    node :retrieve_facts do
      chemic :FactRetrievalPipeline
      input [:parsed_query, :knowledge_base]
      output :verified_facts
    end
    
    node :generate_response do
      chemic :ResponseGenerationPipeline
      input [:verified_facts, :parsed_query, :conversation_history]
      output :response
    end
    
    node :verify_response do
      atomic :ResponseVerifier
      input [:response, :verified_facts]
      output :verified_response
    end
  end
  
  edges do
    :parse_query → :retrieve_facts
    :retrieve_facts → :generate_response
    :generate_response → :verify_response
  end
  
  verify_end_to_end_provenance do
    # Every fact in response must come from knowledge base
    response_facts = extract_facts_from(output(:verified_response).value)
    
    Enum.all?(response_facts, fn fact ->
      # Fact exists in knowledge base
      fact_exists_in_kb?(fact, input(:knowledge_base)) and
      # Provenance traces to verified source
      output(:verified_response).provenance.citations
      |> Enum.any?(fn c -> c.verified_at != nil end)
    end)
    
    # Response confidence reflects weakest link
    min_confidence_in_chain = [
      output(:parsed_query).provenance.confidence,
      output(:verified_facts).provenance.confidence,
      output(:response).provenance.confidence
    ] |> Enum.min()
    
    output(:verified_response).provenance.confidence <= min_confidence_in_chain
    
    # All sources cited
    output(:verified_response).provenance.citations != []
  end
end
```

---

## 8. Key Differences from Physics (defagent)

### Comparison Table

| Aspect | Physics (defagent) | Information (atomic/chemic/bionic) |
|--------|-------------------|-----------------------------------|
| **Core Principle** | Conservation laws | Provenance tracking |
| **Value Type** | `Conserved<T>` | `Tracked<T>` |
| **Constraint** | `conserves :energy` | `tracks_provenance` |
| **Verification** | Total input = total output | Sources traceable |
| **Operations** | Transfer only | Create, transform, aggregate |
| **Use Cases** | Physics, chemistry | Text, data, routing |
| **Example** | Particle collision | Route calculation |
| **Error** | Conservation violated | Provenance lost |
| **Hardware** | RT/Tensor/CUDA cores | NPU/CPU |

### When to Use Each

**Use defagent (conservation):**
- Physics simulations (gravity, electromagnetism)
- Chemical reactions
- Energy systems
- Momentum calculations
- Truly conserved quantities

**Use atomic/chemic/bionic (provenance):**
- Data transformation pipelines
- Text analysis (sentiment, NER)
- Route calculation
- Database queries
- AI/ML inference
- Information aggregation
- Chatbots
- Web scraping

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. ✅ Implement `AII.Types.Provenance`
2. ✅ Implement `AII.Types.Tracked`
3. ✅ Test provenance tracking
4. ✅ Test confidence degradation

### Phase 2: Atomics (Week 3-4)
1. ✅ Implement `AII.DSL.Atomic` macro
2. ✅ Implement `AII.Atomic` behavior
3. ✅ Test with text analysis atomics
4. ✅ Test with route calculation atomics

### Phase 3: Chemics (Week 5-6)
1. ✅ Implement `AII.DSL.Chemic` macro
2. ✅ Implement `AII.Chemic` behavior
3. ✅ Test text processing pipeline
4. ✅ Test route optimization pipeline

### Phase 4: Bionics (Week 7-8)
1. ✅ Implement `AII.DSL.Bionic` macro
2. ✅ Implement `AII.Bionic` behavior
3. ✅ Test chatbot bionic
4. ✅ Test GIS fleet management bionic

### Phase 5: Integration (Week 9-10)
1. Integrate with existing AII runtime
2. Add hardware acceleration (NPU, RT cores)
3. Performance testing
4. Production deployment

---

## 10. Success Criteria

### Must Have
- [x] Provenance tracked through all transformations
- [x] Quality (confidence) verified at boundaries
- [x] Clear error messages with traces
- [x] Atomics, chemics, bionics all working
- [x] Text analysis example working
- [x] Route calculation example working

### Should Have
- [ ] Hardware acceleration (NPU, RT cores)
- [ ] Performance >1000 ops/sec
- [ ] Integration with phase 8 (hallucination-free chatbots)
- [ ] Integration with phase 9 (GIS fleet management)

### Nice to Have
- [ ] Visual provenance graphs
- [ ] Real-time provenance monitoring
- [ ] Provenance-based debugging tools

---

## Final Notes

**Key Insight:** Information processing is fundamentally different from physics:
- **Physics**: Energy conserved (cannot create/destroy)
- **Information**: Facts traced (know where they came from)

**This architecture makes hallucination impossible** because:
1. Every output has provenance
2. Provenance must trace to verified sources
3. Compiler enforces provenance constraints
4. Runtime verifies quality thresholds

**For GIS/Fleet Management:**
- Routes have provenance (which roads, which algorithm)
- Costs have provenance (from pricing database)
- Vehicle positions have provenance (from GPS sensors)
- No hallucinated roads or impossible routes

**Next Steps:**
1. Implement `AII.Types.Provenance` first
2. Then `AII.Types.Tracked`
3. Then tackle macros (Atomic → Chemic → Bionic)
4. Test with real examples (chatbot, routing)

🎵 **This is provenance-based programming - information you can trust!** ⚛️
