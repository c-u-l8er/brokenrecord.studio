defmodule AII.Examples.AtomicChemicBionicExample do
  @moduledoc """
  Example demonstrating the Atomic, Chemic, and Bionic DSLs with provenance tracking.

  This example defines:
  - An atomic that doubles a number with provenance tracking
  - A chemic that doubles twice (composes two atomics)
  - A bionic that orchestrates the chemic

  It also includes tests to verify provenance and execution.
  """

  require AII.DSL.Atomic
  require AII.DSL.Chemic
  require AII.DSL.Bionic

  import AII.DSL.Atomic
  import AII.DSL.Chemic
  import AII.DSL.Bionic

  # Define an atomic that doubles a number with provenance tracking
  defatomic Double do
    input(:value, :number)

    requires_quality(0.8)

    kernel do
      doubled = inputs[:value].value * 2

      # Create tracked output with provenance
      result =
        AII.Types.Tracked.transform(
          inputs[:value],
          :Double,
          :multiply,
          doubled,
          # Slight degradation
          inputs[:value].provenance.confidence * 0.95,
          %{factor: 2}
        )

      %{result: result}
    end
  end

  # Text Analysis Atomic: analyzes text for sentiment
  defatomic SentimentAnalysis do
    input(:text, :string)

    output(:sentiment, :float, confidence_degradation: 0.1)
    output(:subjectivity, :float, confidence_degradation: 0.05)

    requires_quality(0.8)

    kernel do
      # Analyze polarity (-1 to 1, negative to positive)
      # Simple keyword-based polarity analysis
      positive_words = ["good", "great", "excellent", "awesome", "love", "happy"]
      negative_words = ["bad", "terrible", "awful", "hate", "sad", "angry"]

      words = String.downcase(inputs[:text].value) |> String.split()

      positive_count = Enum.count(words, &(&1 in positive_words))
      negative_count = Enum.count(words, &(&1 in negative_words))

      total_sentiment_words = positive_count + negative_count

      polarity =
        if total_sentiment_words == 0 do
          0.0
        else
          (positive_count - negative_count) / total_sentiment_words
        end

      # Analyze subjectivity (0 to 1, objective to subjective)
      # Simple heuristic: longer sentences are more subjective
      sentences = String.split(inputs[:text].value, ~r/[.!?]/, trim: true)

      avg_sentence_length =
        Enum.sum(Enum.map(sentences, &String.length/1)) / max(length(sentences), 1)

      # Normalize to 0-1 range (assuming 20 chars avg is subjective)
      subjectivity = min(avg_sentence_length / 20.0, 1.0)

      # Create tracked outputs with provenance
      sentiment_output =
        AII.Types.Tracked.transform(
          inputs[:text],
          :SentimentAnalysis,
          :sentiment_analysis,
          polarity,
          inputs[:text].provenance.confidence * 0.9,
          %{analysis_type: :polarity}
        )

      subjectivity_output =
        AII.Types.Tracked.transform(
          inputs[:text],
          :SentimentAnalysis,
          :subjectivity_analysis,
          subjectivity,
          inputs[:text].provenance.confidence * 0.95,
          %{analysis_type: :subjectivity}
        )

      %{sentiment: sentiment_output, subjectivity: subjectivity_output}
    end
  end

  # Database Query Atomic: executes SQL query and returns results
  defatomic DatabaseQuery do
    input(:query, :string)
    input(:connection_params, :map, required: false)

    output(:results, :list, confidence_degradation: 0.05)

    requires_quality(0.9)

    kernel do
      # Mock SQL execution - parse simple SELECT queries
      results =
        case String.upcase(String.trim(inputs[:query].value)) do
          "SELECT * FROM USERS" ->
            [
              %{id: 1, name: "Alice", email: "alice@example.com"},
              %{id: 2, name: "Bob", email: "bob@example.com"}
            ]

          "SELECT COUNT(*) FROM USERS" ->
            [%{count: 2}]

          _ ->
            # For other queries, return empty list with warning
            IO.puts("Mock DB: Unsupported query: #{inputs[:query].value}")
            []
        end

      # Create tracked output with provenance
      results_output =
        AII.Types.Tracked.transform(
          inputs[:query],
          :DatabaseQuery,
          :sql_execution,
          results,
          inputs[:query].provenance.confidence * 0.95,
          %{query: inputs[:query].value, row_count: length(results)}
        )

      %{results: results_output}
    end
  end

  # Route Calculation Atomic: calculates route using Dijkstra's algorithm
  defatomic RouteCalculation do
    input(:network, :map)
    input(:start_node, :atom)
    input(:end_node, :atom)

    output(:route, :list, confidence_degradation: 0.1)

    requires_quality(0.85)

    kernel do
      # Mock route calculation - return a simple path
      route = [inputs[:start_node].value, :b, inputs[:end_node].value]

      # Create tracked output with provenance
      route_output =
        AII.Types.Tracked.transform(
          inputs[:network],
          :RouteCalculation,
          :dijkstra_routing,
          route,
          inputs[:network].provenance.confidence * 0.9,
          %{
            start: inputs[:start_node].value,
            end: inputs[:end_node].value,
            route_length: length(route)
          }
        )

      %{route: route_output}
    end
  end

  # Define a chemic: composes two doubles
  defchemic DoubleTwice do
    atomic(:first, Atomic.Double)
    atomic(:second, Atomic.Double)

    bonds do
      bond(:first, :second)
    end

    tracks_pipeline_provenance do
      # Output must trace back to input
      output = outputs[:result]
      input = inputs[:value]

      output.provenance.source_id == input.provenance.source_id and
        length(output.provenance.transformation_chain) == 2
    end
  end

  # Text Processing Pipeline Chemic: analyzes text sentiment and subjectivity
  defchemic TextProcessingPipeline do
    atomic(:sentiment_analyzer, Atomic.SentimentAnalysis)

    # No bonds needed since only one atomic
  end

  # Route Optimization Pipeline Chemic: calculates optimal route
  defchemic RouteOptimizationPipeline do
    atomic(:route_calculator, Atomic.RouteCalculation)

    # No bonds needed since only one atomic
  end

  # Define a bionic: orchestrates the double_twice chemic
  defbionic DoubleBionic do
    inputs do
      stream(:value, type: :number)
    end

    dag do
      node :process do
        chemic(Chemic.DoubleTwice)
      end
    end

    verify_end_to_end_provenance do
      # End-to-end verification
      output = outputs[:result]
      input = inputs[:value]

      output.provenance.source_id == input.provenance.source_id and
        length(output.provenance.transformation_chain) == 2
    end
  end

  # Chatbot Bionic: processes user messages with sentiment analysis
  defbionic Chatbot do
    inputs do
      stream(:message, type: :string)
    end

    dag do
      node :analyze_sentiment do
        chemic(Chemic.TextProcessingPipeline)
      end
    end

    verify_end_to_end_provenance do
      # Ensure sentiment analysis traces back to original message
      sentiment = outputs[:sentiment]
      subjectivity = outputs[:subjectivity]
      input = inputs[:message]

      sentiment.provenance.source_id == input.provenance.source_id and
        subjectivity.provenance.source_id == input.provenance.source_id
    end
  end

  # Test function with provenance verification
  def run_example do
    IO.puts("Running Atomic Chemic Bionic Example with Provenance")

    # Test 1: Double Atomic, Chemic, Bionic
    IO.puts("\n=== Test 1: Double Operations ===")

    double_input =
      AII.Types.Tracked.new(5.0, "double_input", :user_input,
        confidence: 1.0,
        metadata: %{user: "test"}
      )

    IO.puts("Input: #{double_input.value}")

    # Test atomic directly
    {:ok, atomic_outputs} = Atomic.Double.execute(%{value: double_input})
    IO.puts("Atomic outputs: #{inspect(atomic_outputs)}")

    # Verify atomic provenance
    atomic_result = atomic_outputs[:result]

    IO.puts(
      "Atomic provenance: source_id=#{atomic_result.provenance.source_id}, transformations=#{length(atomic_result.provenance.transformation_chain)}"
    )

    # Test chemic directly
    {:ok, chemic_outputs} = Chemic.DoubleTwice.execute(%{value: double_input})
    IO.puts("Chemic outputs: #{inspect(chemic_outputs)}")

    # Verify chemic provenance
    chemic_result = chemic_outputs[:result]

    IO.puts(
      "Chemic provenance: source_id=#{chemic_result.provenance.source_id}, transformations=#{length(chemic_result.provenance.transformation_chain)}"
    )

    # Run the bionic
    case Bionic.DoubleBionic.run(%{value: double_input}) do
      {:ok, outputs} ->
        IO.puts("Bionic outputs: #{inspect(outputs)}")
        result = outputs[:result]

        if result do
          IO.puts("Output value: #{result.value}")
          IO.puts("Expected: 20.0 (5 * 2 * 2)")

          IO.puts(
            "Bionic provenance: source_id=#{result.provenance.source_id}, transformations=#{length(result.provenance.transformation_chain)}, confidence=#{result.provenance.confidence}"
          )
        else
          IO.puts("✗ Result is nil")
        end
    end

    # Test 2: Sentiment Analysis Atomic
    IO.puts("\n=== Test 2: Sentiment Analysis ===")

    text_input =
      AII.Types.Tracked.new("I love this great product! It's awesome.", "text_input", :user_input,
        confidence: 1.0
      )

    {:ok, sentiment_outputs} = Atomic.SentimentAnalysis.execute(%{text: text_input})
    IO.puts("Sentiment outputs: #{inspect(sentiment_outputs)}")

    sentiment = sentiment_outputs[:sentiment]
    subjectivity = sentiment_outputs[:subjectivity]
    IO.puts("Sentiment: #{sentiment.value}, Subjectivity: #{subjectivity.value}")

    # Test 3: Database Query Atomic
    IO.puts("\n=== Test 3: Database Query ===")

    query_input =
      AII.Types.Tracked.new("SELECT * FROM USERS", "query_input", :user_input, confidence: 1.0)

    {:ok, db_outputs} = Atomic.DatabaseQuery.execute(%{query: query_input})
    IO.puts("DB outputs: #{inspect(db_outputs)}")

    results = db_outputs[:results]
    IO.puts("Query results: #{inspect(results.value)}")

    # Test 4: Route Calculation Atomic
    IO.puts("\n=== Test 4: Route Calculation ===")

    network_input =
      AII.Types.Tracked.new(
        %{a: [b: 1, c: 4], b: [a: 1, c: 2, d: 5], c: [a: 4, b: 2, d: 1], d: [b: 5, c: 1]},
        "network_input",
        :database_query,
        confidence: 0.95
      )

    start_input = AII.Types.Tracked.new(:a, "start_input", :user_input, confidence: 1.0)
    end_input = AII.Types.Tracked.new(:d, "end_input", :user_input, confidence: 1.0)

    {:ok, route_outputs} =
      Atomic.RouteCalculation.execute(%{
        network: network_input,
        start_node: start_input,
        end_node: end_input
      })

    IO.puts("Route outputs: #{inspect(route_outputs)}")

    route = route_outputs[:route]
    IO.puts("Calculated route: #{inspect(route.value)}")

    # Test 5: Text Processing Pipeline Chemic
    IO.puts("\n=== Test 5: Text Processing Pipeline ===")
    {:ok, text_pipeline_outputs} = Chemic.TextProcessingPipeline.execute(%{text: text_input})
    IO.puts("Text pipeline outputs: #{inspect(text_pipeline_outputs)}")

    # Test 6: Route Optimization Pipeline Chemic
    IO.puts("\n=== Test 6: Route Optimization Pipeline ===")

    {:ok, route_pipeline_outputs} =
      Chemic.RouteOptimizationPipeline.execute(%{
        network: network_input,
        start_node: start_input,
        end_node: end_input
      })

    IO.puts("Route pipeline outputs: #{inspect(route_pipeline_outputs)}")

    # Test 7: Chatbot Bionic
    IO.puts("\n=== Test 7: Chatbot Bionic ===")

    case Bionic.Chatbot.run(%{message: text_input}) do
      {:ok, chatbot_outputs} ->
        IO.puts("Chatbot outputs: #{inspect(chatbot_outputs)}")
        sentiment_out = chatbot_outputs[:sentiment]
        subjectivity_out = chatbot_outputs[:subjectivity]

        IO.puts(
          "Chatbot sentiment: #{sentiment_out.value}, subjectivity: #{subjectivity_out.value}"
        )
    end
  end
end

AII.Examples.AtomicChemicBionicExample.run_example()
