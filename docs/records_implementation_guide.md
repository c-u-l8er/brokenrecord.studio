# BrokenRecord Implementation Guide
## Critical Implementation Details for Records/Playlists/Workflows DSL

**Context:** This doc assumes you have read the other AII architecture documents.  
**Goal:** Implement the hierarchical composition system (Particles → Records → Playlists → Workflows).  
**Audience:** LLM coding assistant

---

## 1. Core Type System

### Critical Types to Implement

```elixir
# lib/aii/types.ex

defmodule AII.Types do
  # Conserved value wrapper (CRITICAL - foundation of everything)
  defmodule Conserved do
    @type t(inner) :: %__MODULE__{
      value: inner,
      source: atom(),
      tracked: boolean(),
      lineage: [atom()]  # NEW: Track full provenance chain
    }
    
    defstruct [:value, :source, :tracked, lineage: []]
    
    # ONLY way to create conserved value
    def new(value, source) when is_atom(source) do
      %__MODULE__{
        value: value,
        source: source,
        tracked: true,
        lineage: [source]
      }
    end
    
    # ONLY way to move conserved value
    def transfer(from, to, amount) do
      if from.value < amount do
        {:error, :insufficient_value}
      else
        new_from = %{from | 
          value: from.value - amount,
          lineage: from.lineage ++ [:transfer]
        }
        new_to = %{to | 
          value: to.value + amount,
          lineage: to.lineage ++ from.lineage
        }
        {:ok, new_from, new_to}
      end
    end
    
    # Check if two conserved values have compatible lineage
    def compatible_lineage?(c1, c2) do
      MapSet.intersection(
        MapSet.new(c1.lineage),
        MapSet.new(c2.lineage)
      ) != MapSet.new()
    end
  end
  
  # Particle type (base unit)
  defmodule Particle do
    defstruct [
      :id,
      :position,
      :velocity, 
      :mass,
      :information,  # Conserved<Float>
      :owner,        # :record_name or :playlist_name
      :metadata
    ]
  end
end
```

---

## 2. Record DSL Implementation

### Critical Macro: `defrecord`

```elixir
# lib/aii/dsl/record.ex

defmodule AII.DSL.Record do
  defmacro defrecord(name, opts \\ [], do: block) do
    quote do
      defmodule unquote(:"AII.Records.#{name |> Macro.camelize()}") do
        use AII.Record
        
        Module.put_attribute(__MODULE__, :record_name, unquote(name))
        Module.put_attribute(__MODULE__, :atomic_number, unquote(opts[:atomic_number]))
        Module.put_attribute(__MODULE__, :record_type, unquote(opts[:type] || :unknown))
        
        unquote(block)
        
        # Auto-generate functions after block is evaluated
        @before_compile AII.Record
      end
    end
  end
  
  # Kernel macro - defines transformation logic
  defmacro kernel(do: block) do
    quote do
      def kernel_function(record_state, inputs) do
        # Make inputs available in scope
        var!(inputs) = inputs
        var!(record) = record_state
        
        unquote(block)
      end
    end
  end
  
  # Input declaration
  defmacro input(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :inputs, {
        unquote(name),
        unquote(opts[:type]),
        unquote(opts[:count] || :single)
      })
    end
  end
  
  # State declaration (internal particles)
  defmacro state(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :state_fields, {
        unquote(name),
        unquote(opts[:type]),
        unquote(opts[:conserves] || false)
      })
    end
  end
  
  # Conservation constraint
  defmacro conserves(quantity, do: block) do
    quote do
      Module.put_attribute(__MODULE__, :conservation_laws, {
        unquote(quantity),
        fn inputs, outputs -> unquote(block) end
      })
    end
  end
  
  # Transform block - the actual computation
  defmacro transform(do: block) do
    quote do
      def transform_function(record_state, inputs) do
        unquote(block)
      end
    end
  end
  
  # Accelerator hint
  defmacro accelerator(type) do
    quote do
      Module.put_attribute(__MODULE__, :accelerator_hint, unquote(type))
    end
  end
end
```

### Critical: Record Behavior Module

```elixir
# lib/aii/record.ex

defmodule AII.Record do
  @callback execute(record_state :: term(), inputs :: map()) :: 
    {:ok, record_state :: term(), outputs :: map()} | {:error, term()}
  
  defmacro __using__(_opts) do
    quote do
      @behaviour AII.Record
      
      import AII.DSL.Record
      
      Module.register_attribute(__MODULE__, :inputs, accumulate: true)
      Module.register_attribute(__MODULE__, :state_fields, accumulate: true)
      Module.register_attribute(__MODULE__, :conservation_laws, accumulate: true)
      Module.register_attribute(__MODULE__, :accelerator_hint, [])
      
      # Default implementation
      def execute(record_state, inputs) do
        # 1. Verify input conservation
        :ok = verify_input_conservation(inputs)
        
        # 2. Run kernel
        {updated_state, outputs} = kernel_function(record_state, inputs)
        
        # 3. Verify output conservation
        :ok = verify_output_conservation(inputs, outputs)
        
        {:ok, updated_state, outputs}
      end
      
      defp verify_input_conservation(inputs) do
        for {quantity, check_fn} <- @conservation_laws do
          unless check_fn.(inputs, nil) do
            raise AII.ConservationViolation, """
            Record #{@record_name} input violated conservation of #{quantity}
            """
          end
        end
        :ok
      end
      
      defp verify_output_conservation(inputs, outputs) do
        for {quantity, check_fn} <- @conservation_laws do
          unless check_fn.(inputs, outputs) do
            raise AII.ConservationViolation, """
            Record #{@record_name} output violated conservation of #{quantity}
            Inputs: #{inspect(inputs)}
            Outputs: #{inspect(outputs)}
            """
          end
        end
        :ok
      end
    end
  end
  
  defmacro __before_compile__(_env) do
    quote do
      # Generate metadata function
      def __record_metadata__ do
        %{
          name: @record_name,
          atomic_number: @atomic_number,
          type: @record_type,
          inputs: @inputs,
          state_fields: @state_fields,
          conservation_laws: @conservation_laws,
          accelerator: @accelerator_hint
        }
      end
    end
  end
end
```

---

## 3. Playlist DSL Implementation

### Critical Macro: `defplaylist`

```elixir
# lib/aii/dsl/playlist.ex

defmodule AII.DSL.Playlist do
  defmacro defplaylist(name, opts \\ [], do: block) do
    quote do
      defmodule unquote(:"AII.Playlists.#{name |> Macro.camelize()}") do
        use AII.Playlist
        
        Module.put_attribute(__MODULE__, :playlist_name, unquote(name))
        Module.put_attribute(__MODULE__, :element_number, unquote(opts[:element_number]))
        Module.put_attribute(__MODULE__, :element_class, unquote(opts[:class]))
        
        unquote(block)
        
        @before_compile AII.Playlist
      end
    end
  end
  
  # Composition - declare records in this playlist
  defmacro composition(do: block) do
    quote do
      def __composition__ do
        unquote(block)
      end
    end
  end
  
  # Record declaration within composition
  defmacro record(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :records, {
        unquote(name),
        unquote(opts[:type])
      })
    end
  end
  
  # Nested playlist
  defmacro playlist(name, opts \\ []) do
    quote do
      Module.put_attribute(__MODULE__, :sub_playlists, {
        unquote(name),
        unquote(opts[:type])
      })
    end
  end
  
  # Bonds - connections between records
  defmacro bonds(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :bonds, unquote(Macro.escape(block)))
    end
  end
  
  # Bond operator: →
  defmacro left → right do
    quote do
      {unquote(left), unquote(right)}
    end
  end
end
```

### Critical: Playlist Execution Engine

```elixir
# lib/aii/playlist.ex

defmodule AII.Playlist do
  defmacro __using__(_opts) do
    quote do
      import AII.DSL.Playlist
      
      Module.register_attribute(__MODULE__, :records, accumulate: true)
      Module.register_attribute(__MODULE__, :sub_playlists, accumulate: true)
      Module.register_attribute(__MODULE__, :bonds, [])
      
      # Execute playlist
      def execute(playlist_state, inputs) do
        # 1. Parse bonds into DAG
        dag = parse_bonds_to_dag(@bonds)
        
        # 2. Topological sort
        execution_order = topological_sort(dag)
        
        # 3. Execute each node in order
        {updated_state, outputs} = 
          Enum.reduce(execution_order, {playlist_state, inputs}, 
            fn node, {state, data} ->
              execute_node(state, data, node)
            end)
        
        # 4. Verify playlist-level conservation
        verify_conservation(inputs, outputs)
        
        {:ok, updated_state, outputs}
      end
      
      defp execute_node(state, data, node_name) do
        # Get record module
        {^node_name, record_type} = 
          Enum.find(@records, fn {name, _} -> name == node_name end)
        
        record_module = AII.Records[record_type]
        
        # Get record state
        record_state = Map.get(state.records, node_name)
        
        # Get inputs for this record from data flow
        node_inputs = get_node_inputs(data, node_name)
        
        # Execute record
        {:ok, updated_record, outputs} = 
          record_module.execute(record_state, node_inputs)
        
        # Update state
        updated_state = 
          put_in(state.records[node_name], updated_record)
        
        # Update data flow
        updated_data = Map.put(data, node_name, outputs)
        
        {updated_state, updated_data}
      end
      
      defp parse_bonds_to_dag(bonds_block) do
        # CRITICAL: Parse AST of bonds block into graph structure
        # bonds_block is like: {:→, meta, [left, right]}
        # Return: %{node => [dependencies]}
        
        bonds_block
        |> extract_arrows()
        |> build_adjacency_list()
      end
      
      defp topological_sort(dag) do
        # CRITICAL: Kahn's algorithm for topological sort
        # Must detect cycles and raise error
        
        AII.Graph.topological_sort(dag)
      rescue
        AII.Graph.CycleDetected -> 
          raise AII.PlaylistError, "Cycle detected in bonds - DAG required"
      end
    end
  end
end
```

---

## 4. Workflow DSL Implementation

### Critical Macro: `defworkflow`

```elixir
# lib/aii/dsl/workflow.ex

defmodule AII.DSL.Workflow do
  defmacro defworkflow(name, opts \\ [], do: block) do
    quote do
      defmodule unquote(:"AII.Workflows.#{name |> Macro.camelize()}") do
        use AII.Workflow
        
        Module.put_attribute(__MODULE__, :workflow_name, unquote(name))
        
        unquote(block)
        
        @before_compile AII.Workflow
      end
    end
  end
  
  # DAG structure
  defmacro dag(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :dag_block, unquote(Macro.escape(block)))
    end
  end
  
  # Node definition
  defmacro node(name, do: block) do
    quote do
      Module.put_attribute(__MODULE__, :nodes, {
        unquote(name),
        unquote(Macro.escape(block))
      })
    end
  end
  
  # Edges between nodes
  defmacro edges(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :edges, unquote(Macro.escape(block)))
    end
  end
  
  # Input/output declarations
  defmacro inputs(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :workflow_inputs, 
        unquote(Macro.escape(block)))
    end
  end
  
  defmacro outputs(do: block) do
    quote do
      Module.put_attribute(__MODULE__, :workflow_outputs, 
        unquote(Macro.escape(block)))
    end
  end
end
```

### Critical: Workflow Execution Engine

```elixir
# lib/aii/workflow.ex

defmodule AII.Workflow do
  defmacro __using__(_opts) do
    quote do
      import AII.DSL.Workflow
      
      Module.register_attribute(__MODULE__, :nodes, accumulate: true)
      Module.register_attribute(__MODULE__, :edges, [])
      Module.register_attribute(__MODULE__, :dag_block, [])
      
      # Main entry point
      def run(inputs, opts \\ []) do
        # 1. Build execution plan
        plan = build_execution_plan()
        
        # 2. Initialize state
        initial_state = initialize_workflow_state()
        
        # 3. Execute plan
        {final_state, outputs} = 
          execute_plan(plan, initial_state, inputs, opts)
        
        # 4. Verify workflow conservation
        verify_workflow_conservation(inputs, outputs)
        
        # 5. Return outputs (and trace if requested)
        case opts[:trace] do
          true -> {:ok, outputs, extract_trace(final_state)}
          _ -> {:ok, outputs}
        end
      end
      
      defp build_execution_plan do
        # Parse DAG structure
        dag = parse_dag(@dag_block, @nodes, @edges)
        
        # Topological sort
        AII.Graph.topological_sort(dag)
      end
      
      defp execute_plan(plan, state, inputs, opts) do
        Enum.reduce(plan, {state, inputs}, fn node_name, {st, data} ->
          # Execute node (playlist)
          {updated_st, node_outputs} = 
            execute_workflow_node(st, data, node_name, opts)
          
          # Update data flow
          updated_data = Map.put(data, node_name, node_outputs)
          
          # Log if tracing
          if opts[:trace] do
            log_execution(node_name, data, node_outputs)
          end
          
          {updated_st, updated_data}
        end)
      end
      
      defp execute_workflow_node(state, data, node_name, opts) do
        # Get node configuration
        {^node_name, node_config} = 
          Enum.find(@nodes, fn {n, _} -> n == node_name end)
        
        # Extract playlist from node config
        playlist_module = extract_playlist_module(node_config)
        
        # Get playlist state
        playlist_state = Map.get(state.playlists, node_name)
        
        # Get inputs for this node
        node_inputs = gather_node_inputs(data, node_config)
        
        # Execute playlist
        {:ok, updated_playlist, outputs} = 
          playlist_module.execute(playlist_state, node_inputs)
        
        # Update state
        updated_state = 
          put_in(state.playlists[node_name], updated_playlist)
        
        {updated_state, outputs}
      end
      
      defp verify_workflow_conservation(inputs, outputs) do
        # CRITICAL: Overall workflow must not create information
        input_info = AII.Conservation.total_information(inputs)
        output_info = AII.Conservation.total_information(outputs)
        
        if output_info > input_info + @tolerance do
          raise AII.ConservationViolation, """
          Workflow #{@workflow_name} created information:
            Inputs:  #{input_info}
            Outputs: #{output_info}
            Created: #{output_info - input_info}
          """
        end
        
        :ok
      end
    end
  end
end
```

---

## 5. Conservation Verification (CRITICAL)

```elixir
# lib/aii/conservation.ex

defmodule AII.Conservation do
  @tolerance 0.0001
  
  # Calculate total information in a data structure
  def total_information(data) when is_map(data) do
    data
    |> Map.values()
    |> Enum.map(&extract_information/1)
    |> Enum.sum()
  end
  
  defp extract_information(%AII.Types.Conserved{value: v}), do: v
  defp extract_information(%AII.Types.Particle{information: info}), 
    do: info.value
  defp extract_information(list) when is_list(list),
    do: Enum.sum(Enum.map(list, &extract_information/1))
  defp extract_information(_), do: 0.0
  
  # Verify conservation between two states
  def verify(before, after_state, opts \\ []) do
    tolerance = opts[:tolerance] || @tolerance
    
    before_info = total_information(before)
    after_info = total_information(after_state)
    
    diff = abs(before_info - after_info)
    
    if diff > tolerance do
      {:error, {:conservation_violated, 
        before: before_info, 
        after: after_info, 
        diff: diff}}
    else
      :ok
    end
  end
  
  # Compile-time symbolic verification (ADVANCED - optional)
  def symbolic_verify(ast) do
    # Analyze AST to prove conservation symbolically
    # This is hard - start with runtime checks
    # Future: constraint solver integration
    :not_implemented
  end
end
```

---

## 6. Graph Utilities (CRITICAL for DAG execution)

```elixir
# lib/aii/graph.ex

defmodule AII.Graph do
  defmodule CycleDetected do
    defexception [:message]
  end
  
  # Kahn's algorithm for topological sort
  def topological_sort(graph) do
    # graph is %{node => [dependencies]}
    
    # Find nodes with no dependencies
    no_deps = find_nodes_with_no_deps(graph)
    
    # Recursive sort
    do_topo_sort(graph, no_deps, [], MapSet.new())
  end
  
  defp do_topo_sort(graph, [], sorted, visited) do
    # Check if all nodes visited
    if MapSet.size(visited) == map_size(graph) do
      Enum.reverse(sorted)
    else
      # Cycle detected - some nodes not reachable
      raise CycleDetected, 
        "Cycle detected in graph - remaining: #{inspect(Map.keys(graph) -- MapSet.to_list(visited))}"
    end
  end
  
  defp do_topo_sort(graph, [node | rest], sorted, visited) do
    if MapSet.member?(visited, node) do
      do_topo_sort(graph, rest, sorted, visited)
    else
      # Add node to sorted list
      new_sorted = [node | sorted]
      new_visited = MapSet.put(visited, node)
      
      # Find newly available nodes
      newly_available = find_newly_available(graph, new_visited)
      
      do_topo_sort(graph, rest ++ newly_available, new_sorted, new_visited)
    end
  end
  
  defp find_nodes_with_no_deps(graph) do
    Enum.filter(graph, fn {_node, deps} ->
      deps == [] or Enum.all?(deps, &is_input?/1)
    end)
    |> Enum.map(fn {node, _} -> node end)
  end
  
  defp find_newly_available(graph, visited) do
    Enum.filter(graph, fn {node, deps} ->
      not MapSet.member?(visited, node) and
      Enum.all?(deps, &MapSet.member?(visited, &1))
    end)
    |> Enum.map(fn {node, _} -> node end)
  end
  
  # Build adjacency list from bonds/edges
  def build_adjacency_list(edges) do
    # edges is list of {from, to} tuples
    Enum.reduce(edges, %{}, fn {from, to}, acc ->
      Map.update(acc, to, [from], fn deps -> [from | deps] end)
    end)
  end
end
```

---

## 7. Hardware Dispatcher Integration

```elixir
# lib/aii/hardware_dispatcher.ex

defmodule AII.HardwareDispatcher do
  # Map record to optimal hardware
  def dispatch(record_module) do
    metadata = record_module.__record_metadata__()
    
    # Check explicit accelerator hint
    case metadata.accelerator do
      :rt_cores -> :rt_cores
      :tensor_cores -> :tensor_cores
      :npu -> :npu
      :cuda_cores -> :cuda_cores
      nil -> infer_accelerator(metadata)
    end
  end
  
  defp infer_accelerator(metadata) do
    # Infer from record type and operations
    cond do
      has_spatial_query?(metadata) -> :rt_cores
      has_matrix_op?(metadata) -> :tensor_cores
      has_learned_model?(metadata) -> :npu
      true -> :cuda_cores
    end
  end
  
  # Generate hardware-specific code
  def generate_code(record_module, accelerator) do
    case accelerator do
      :rt_cores -> 
        AII.Codegen.Vulkan.generate_ray_query(record_module)
      :tensor_cores -> 
        AII.Codegen.Vulkan.generate_tensor_op(record_module)
      :npu -> 
        AII.Codegen.NPU.generate_inference(record_module)
      :cuda_cores -> 
        AII.Codegen.CUDA.generate_kernel(record_module)
    end
  end
end
```

---

## 8. Example Usage (For Testing)

```elixir
# test/examples/simple_workflow_test.exs

defmodule AII.Examples.SimpleWorkflowTest do
  use ExUnit.Case
  
  # Define a simple record
  defrecord :double do
    @atomic_number 1
    
    kernel do
      input :value, type: :float
      
      conserves :information do
        inputs[:value].info == outputs[:result].info
      end
      
      transform do
        result = Conserved.new(
          inputs[:value].value * 2,
          :double
        )
        
        %{result: result}
      end
    end
  end
  
  # Define a playlist
  defplaylist :double_twice do
    @element_number 100
    
    composition do
      record :first, type: :double
      record :second, type: :double
    end
    
    bonds do
      input(:x) → :first
      :first → :second
      :second → output(:result)
    end
  end
  
  # Define a workflow
  defworkflow :double_workflow do
    dag do
      node :process do
        playlist :double_twice
        input :number
      end
    end
    
    conserves :information
  end
  
  test "workflow executes with conservation" do
    input = Conserved.new(5.0, :user_input)
    
    {:ok, result} = AII.Workflows.DoubleWorkflow.run(%{number: input})
    
    # 5 * 2 * 2 = 20
    assert result.result.value == 20.0
    
    # Conservation verified
    assert result.result.info == input.info
  end
end
```

---

## 9. Critical Implementation Order

### Phase 1: Foundation (Week 1-2)
1. **Conserved type** - Must be bulletproof
2. **Particle type** - Base unit
3. **Conservation checker** - verify/2 function
4. **Graph utilities** - Topological sort

### Phase 2: Records (Week 3-4)
1. **defrecord macro** - Parse and generate
2. **Record behavior** - execute/2 
3. **Kernel execution** - Transform with conservation
4. **Test with simple records** (add, multiply)

### Phase 3: Playlists (Week 5-6)
1. **defplaylist macro** - Parse bonds
2. **Bond parsing** - AST → DAG
3. **Playlist execution** - execute_node/3
4. **Test with simple playlists**

### Phase 4: Workflows (Week 7-8)
1. **defworkflow macro** - Parse DAG
2. **Workflow execution** - run/2
3. **Edge verification** - Conservation at boundaries
4. **Test end-to-end**

### Phase 5: Hardware (Week 9-10)
1. **Hardware dispatcher** - dispatch/1
2. **Code generation stubs** - generate_code/2
3. **Integration with records**
4. **Performance testing**

---

## 10. Critical Edge Cases to Handle

### 1. Parallel Branches in Playlists
```elixir
# This must work:
bonds do
  input(:x) → :branch1
  input(:x) → :branch2
  [:branch1, :branch2] → :combine
  :combine → output(:result)
end

# Conservation: total(branch1 + branch2) == input(x)
```

### 2. Conditional Execution
```elixir
# Records can conditionally emit:
transform do
  if condition do
    emit(result1)
  else
    emit(result2)
  end
  # Conservation still verified!
end
```

### 3. Loops (Special Case)
```elixir
# Normally forbidden (DAG only)
# But iterative workflows need special handling:
defworkflow :iterative do
  loop max_iterations: 10 do
    node :refine
    until :refine.output.quality > threshold
  end
end
```

### 4. Error Propagation
```elixir
# Conservation violations should:
# 1. Include full trace
# 2. Show exact location
# 3. Provide debugging info
```

---

## 11. Testing Strategy

### Unit Tests
- Each macro independently
- Conservation checker with edge cases
- Graph utilities (cycles, diamonds, etc)

### Integration Tests  
- Simple record → playlist → workflow
- Conservation across boundaries
- Error messages quality

### Property Tests
- Conservation always holds
- No information created
- DAG properties maintained

### Performance Tests
- Large playlists (100+ records)
- Deep workflows (10+ layers)
- Hardware dispatch overhead

---

## 12. What NOT to Implement Yet

### Skip These Initially:
1. ❌ Symbolic verification (too complex)
2. ❌ Visual DAG editor (separate project)
3. ❌ Hardware code generation (use stubs)
4. ❌ Distributed execution (later)
5. ❌ Optimize for performance (correctness first)

### Focus On:
1. ✅ Correct macro expansion
2. ✅ Conservation verification
3. ✅ Clear error messages
4. ✅ Clean DSL syntax
5. ✅ End-to-end examples

---

## 13. Success Criteria

### Must Have:
- [ ] Records execute with conservation ✓
- [ ] Playlists compose records ✓
- [ ] Workflows compose playlists ✓
- [ ] Conservation verified at all boundaries ✓
- [ ] Clear error messages with traces ✓
- [ ] Example chatbot works ✓

### Should Have:
- [ ] Hardware dispatcher functional
- [ ] Parallel execution in playlists
- [ ] Good performance (>1000 particles/sec)

### Nice to Have:
- [ ] Compile-time optimizations
- [ ] Visual trace output
- [ ] Integration with existing AII runtime

---

## 14. Key Files to Create

```
lib/
├── aii/
│   ├── types.ex                 # Conserved, Particle
│   ├── conservation.ex          # verify/2
│   ├── graph.ex                 # topological_sort/1
│   ├── record.ex                # Record behavior
│   ├── playlist.ex              # Playlist behavior
│   ├── workflow.ex              # Workflow behavior
│   ├── hardware_dispatcher.ex   # dispatch/1
│   └── dsl/
│       ├── record.ex            # defrecord macro
│       ├── playlist.ex          # defplaylist macro
│       └── workflow.ex          # defworkflow macro
test/
├── aii/
│   ├── types_test.exs
│   ├── conservation_test.exs
│   ├── graph_test.exs
│   └── integration_test.exs
└── examples/
    ├── simple_records_test.exs
    ├── transformer_test.exs
    └── chatbot_test.exs
```

---

## 15. Final Notes for LLM

**Start with:**
1. Implement `AII.Types.Conserved` first - it's the foundation
2. Then `AII.Conservation.verify/2` - you'll need it everywhere
3. Then `AII.Graph.topological_sort/1` - critical for DAG execution
4. Then tackle macros one at a time (Record → Playlist → Workflow)

**Remember:**
- Conservation is NON-NEGOTIABLE - raise errors liberally
- Clear error messages > performance
- Test each piece before moving on
- The DSL should feel natural to write

**This is the architecture that makes hallucination impossible.** 🎵⚛️

Get started with `AII.Types.Conserved` - it's literally the atom of everything!