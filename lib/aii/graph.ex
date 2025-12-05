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
      deps == []
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
