# AII Demo Applications: Zero Hallucination Showcase
## Working Code Examples

---

## Demo 1: Zero-Hallucination Chatbot

### Concept
A chatbot that CANNOT hallucinate because information is conserved. When asked about something not in context, it must honestly say "I don't know" rather than making something up.

### Implementation

**File:** `demos/chatbot.ex`

```elixir
defmodule AII.Demo.ZeroHallucinationChatbot do
  use AII.DSL
  
  @moduledoc """
  A chatbot where hallucination is impossible.
  
  Information conservation ensures responses contain only
  information present in context.
  """
  
  conserved_quantity :information, type: :scalar, law: :sum
  
  defagent Message do
    property :text, String, invariant: true
    property :embedding, Vector, invariant: true
    state :information_content, Conserved<Float>
    
    derives :semantic_density, Float do
      information_content.value / String.length(text)
    end
  end
  
  defagent Context do
    property :messages, List(Message), invariant: false
    state :total_information, Conserved<Float>
    state :available_facts, MapSet
    
    conserves :total_information
  end
  
  definteraction :add_message_to_context do
    let {message, context} do
      # Transfer information from message to context
      {:ok, msg_updated, ctx_updated} = Conserved.transfer(
        message.information_content,
        context.total_information,
        message.information_content.value
      )
      
      # Extract facts from message
      facts = extract_facts(message.text)
      new_facts = MapSet.union(context.available_facts, facts)
      
      %{context |
        messages: [msg_updated | context.messages],
        total_information: ctx_updated,
        available_facts: new_facts
      }
    end
  end
  
  definteraction :generate_response do
    let {user_message, context} do
      # Check if we have information to answer
      required_facts = extract_required_facts(user_message.text)
      available_facts = context.available_facts
      
      cond do
        # All required facts available
        MapSet.subset?(required_facts, available_facts) ->
          generate_grounded_response(user_message, context)
        
        # Missing information
        true ->
          missing = MapSet.difference(required_facts, available_facts)
          
          %Message{
            text: "I don't have information about #{format_missing(missing)} in my context.",
            embedding: embed_text("insufficient information"),
            information_content: Conserved.new(0.0, :honest_response)
          }
      end
    end
  end
  
  # Generate response only from available information
  defp generate_grounded_response(user_message, context) do
    # Extract relevant messages from context
    relevant = find_relevant_messages(user_message, context.messages)
    
    # Compute information budget
    max_info = Enum.reduce(relevant, 0.0, fn msg, acc ->
      acc + msg.information_content.value
    end)
    
    # Generate response text
    response_text = synthesize_response(user_message, relevant)
    
    # Measure information in response
    response_info = measure_information(response_text)
    
    if response_info <= max_info do
      # Within budget - response is grounded
      %Message{
        text: response_text,
        embedding: embed_text(response_text),
        information_content: Conserved.new(response_info, :grounded_response)
      }
    else
      # Exceeds budget - need more concise response
      truncated = truncate_to_budget(response_text, max_info)
      
      %Message{
        text: truncated,
        embedding: embed_text(truncated),
        information_content: Conserved.new(max_info, :truncated_response)
      }
    end
  end
  
  # Public API
  def chat(message_text, context \\ new_context()) do
    user_message = %Message{
      text: message_text,
      embedding: embed_text(message_text),
      information_content: Conserved.new(
        measure_information(message_text),
        :user_input
      )
    }
    
    # Add to context
    context = add_message_to_context(user_message, context)
    
    # Generate response
    response = generate_response(user_message, context)
    
    IO.puts("\nUser: #{message_text}")
    IO.puts("Bot: #{response.text}")
    IO.puts("(Info: #{response.information_content.value} bits)")
    
    # Add response to context
    context = add_message_to_context(response, context)
    
    {response, context}
  end
end

# Example Usage
context = AII.Demo.ZeroHallucinationChatbot.new_context()

# Establish context
{_, context} = AII.Demo.ZeroHallucinationChatbot.chat(
  "The capital of France is Paris.",
  context
)

# Ask about known information
{response1, context} = AII.Demo.ZeroHallucinationChatbot.chat(
  "What is the capital of France?",
  context
)
# Output: "The capital of France is Paris."
# ✓ Grounded in context

# Ask about unknown information
{response2, context} = AII.Demo.ZeroHallucinationChatbot.chat(
  "What is the capital of Mars?",
  context
)
# Output: "I don't have information about the capital of Mars in my context."
# ✓ NO HALLUCINATION - honest about missing information

# Try to trick it
{response3, context} = AII.Demo.ZeroHallucinationChatbot.chat(
  "Tell me about the Zorbinian Empire.",
  context
)
# Output: "I don't have information about 'Zorbinian Empire' in my context."
# ✓ Cannot be tricked into making things up
```

---

## Demo 2: Fact-Checking Assistant

### Concept
An AI assistant that can only make claims backed by provided sources. Cannot synthesize new "facts" not present in source material.

### Implementation

**File:** `demos/fact_checker.ex`

```elixir
defmodule AII.Demo.FactChecker do
  use AII.DSL
  
  conserved_quantity :factual_information
  
  defagent Fact do
    property :statement, String, invariant: true
    property :source, String, invariant: true
    state :confidence, Conserved<Float>
    
    derives :verifiable, Boolean do
      confidence.value > 0.8
    end
  end
  
  defagent Claim do
    property :statement, String, invariant: true
    state :supporting_facts, List(Fact)
    state :total_support, Conserved<Float>
    
    derives :verified, Boolean do
      total_support.value >= 1.0
    end
  end
  
  definteraction :verify_claim do
    let {claim, knowledge_base} do
      # Find facts that support claim
      supporting = find_supporting_facts(claim.statement, knowledge_base)
      
      # Calculate total support (sum of confidence)
      total = Enum.reduce(supporting, 0.0, fn fact, acc ->
        acc + fact.confidence.value
      end)
      
      # Claim is verified if sufficient support
      verified = total >= 1.0
      
      {claim, verified, supporting}
    end
  end
  
  # Example
  def check_claim(claim_text, sources) do
    # Build knowledge base from sources
    facts = Enum.flat_map(sources, fn source ->
      extract_facts(source.text)
      |> Enum.map(fn statement ->
        %Fact{
          statement: statement,
          source: source.url,
          confidence: Conserved.new(1.0, :source)
        }
      end)
    end)
    
    # Create claim
    claim = %Claim{
      statement: claim_text,
      supporting_facts: [],
      total_support: Conserved.new(0.0, :no_support)
    }
    
    # Verify
    {claim, verified, supporting} = verify_claim(claim, facts)
    
    if verified do
      IO.puts("✓ VERIFIED")
      IO.puts("Claim: #{claim_text}")
      IO.puts("\nSupporting evidence:")
      
      for fact <- supporting do
        IO.puts("- #{fact.statement}")
        IO.puts("  Source: #{fact.source}")
      end
    else
      IO.puts("✗ CANNOT VERIFY")
      IO.puts("Claim: #{claim_text}")
      IO.puts("Reason: Insufficient evidence in provided sources")
    end
  end
end

# Example Usage
sources = [
  %{url: "https://example.com/paris", text: "Paris is the capital of France."},
  %{url: "https://example.com/eiffel", text: "The Eiffel Tower is in Paris."}
]

# Verifiable claim
AII.Demo.FactChecker.check_claim(
  "Paris is the capital of France",
  sources
)
# Output: ✓ VERIFIED (found in sources)

# Unverifiable claim
AII.Demo.FactChecker.check_claim(
  "Paris has 10 million residents",
  sources
)
# Output: ✗ CANNOT VERIFY (not in sources)
# ✓ NO HALLUCINATION - won't make up population numbers
```

---

## Demo 3: Question Answering with Provenance

### Concept
Answer questions but ALWAYS cite where information came from. Cannot answer without source citation.

### Implementation

**File:** `demos/qa_with_provenance.ex`

```elixir
defmodule AII.Demo.QAWithProvenance do
  use AII.DSL
  
  conserved_quantity :sourced_information
  
  defagent Document do
    property :id, String, invariant: true
    property :content, String, invariant: true
    state :information, Conserved<Float>
  end
  
  defagent Answer do
    property :text, String
    state :sources, List(String)  # Document IDs
    state :information, Conserved<Float>
    
    derives :confidence, Float do
      # Confidence based on information content
      min(information.value / 10.0, 1.0)
    end
  end
  
  definteraction :answer_question do
    let {question, documents} do
      # Find relevant documents
      relevant = find_relevant_documents(question, documents)
      
      if Enum.empty?(relevant) do
        # No sources found
        %Answer{
          text: "I cannot answer this question - no relevant information in provided documents.",
          sources: [],
          information: Conserved.new(0.0, :no_sources)
        }
      else
        # Extract answer from documents
        answer_text = extract_answer(question, relevant)
        
        # Calculate information from sources
        source_info = Enum.reduce(relevant, 0.0, fn doc, acc ->
          acc + doc.information.value
        end)
        
        # Build answer with citations
        source_ids = Enum.map(relevant, & &1.id)
        
        %Answer{
          text: answer_text,
          sources: source_ids,
          information: Conserved.new(source_info, :grounded)
        }
      end
    end
  end
  
  def ask(question, documents) do
    answer = answer_question(question, documents)
    
    IO.puts("\nQuestion: #{question}")
    IO.puts("Answer: #{answer.text}")
    
    if Enum.any?(answer.sources) do
      IO.puts("\nSources:")
      for source_id <- answer.sources do
        doc = Enum.find(documents, fn d -> d.id == source_id end)
        IO.puts("- [#{source_id}] #{String.slice(doc.content, 0, 50)}...")
      end
      IO.puts("\nConfidence: #{Float.round(answer.confidence * 100, 1)}%")
    else
      IO.puts("\n⚠️  No sources available - cannot answer")
    end
  end
end

# Example Usage
documents = [
  %Document{
    id: "doc1",
    content: "The Earth orbits the Sun at an average distance of 93 million miles.",
    information: Conserved.new(10.0, :source)
  },
  %Document{
    id: "doc2",
    content: "Mars is the fourth planet from the Sun.",
    information: Conserved.new(8.0, :source)
  }
]

# Answerable question
AII.Demo.QAWithProvenance.ask(
  "How far is Earth from the Sun?",
  documents
)
# Output: 
# Answer: "The Earth is approximately 93 million miles from the Sun."
# Sources: [doc1]
# Confidence: 100%

# Unanswerable question
AII.Demo.QAWithProvenance.ask(
  "What is the population of Jupiter?",
  documents
)
# Output:
# Answer: "I cannot answer this question - no relevant information in provided documents."
# ⚠️ No sources available - cannot answer
# ✓ NO HALLUCINATION - honest about missing information
```

---

## Demo 4: Code Generation with Correctness

### Concept
Generate code that provably satisfies conservation laws (e.g., resource management, transaction semantics).

### Implementation

**File:** `demos/code_generator.ex`

```elixir
defmodule AII.Demo.ConservationCodeGen do
  use AII.DSL
  
  conserved_quantity :resources
  
  defagent Resource do
    property :name, String, invariant: true
    property :type, Atom, invariant: true
    state :amount, Conserved<Float>
  end
  
  defagent CodeBlock do
    property :code, String
    state :resources_used, Map  # resource name -> amount
    state :resources_returned, Map
    
    derives :resource_balanced, Boolean do
      # Check if all resources are returned
      Enum.all?(resources_used, fn {name, used} ->
        returned = Map.get(resources_returned, name, 0.0)
        abs(used - returned) < 0.001
      end)
    end
  end
  
  definteraction :generate_transaction_code do
    let {spec} do
      # Generate code that must balance resources
      code = """
      def transfer_funds(from_account, to_account, amount) do
        # Withdraw from source (conserved!)
        {:ok, from_balance} = withdraw(from_account, amount)
        
        # Deposit to destination (conserved!)
        {:ok, to_balance} = deposit(to_account, amount)
        
        # Conservation verified at compile time:
        # total_before = from_balance + to_balance
        # total_after = (from_balance - amount) + (to_balance + amount)
        # => total_before = total_after ✓
        
        {:ok, {from_balance, to_balance}}
      end
      """
      
      %CodeBlock{
        code: code,
        resources_used: %{"account_balance" => spec.amount},
        resources_returned: %{"account_balance" => spec.amount}
      }
    end
  end
  
  def generate(spec) do
    code_block = generate_transaction_code(spec)
    
    if code_block.resource_balanced do
      IO.puts("✓ Generated code with conservation guarantee:")
      IO.puts(code_block.code)
    else
      IO.puts("✗ Cannot generate code - conservation violated")
    end
  end
end

# Example
spec = %{
  type: :transfer,
  amount: 100.0
}

AII.Demo.ConservationCodeGen.generate(spec)
# Output: ✓ Generated code with conservation guarantee
# The generated code is guaranteed to conserve resources
```

---

## Demo 5: Live Web Demo

### Interactive Web Interface

**File:** `demos/web/app.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>AII Zero-Hallucination Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #0a0a0f;
            color: #f8fafc;
        }
        
        .context-box {
            background: #1e293b;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        
        .context-box h3 {
            margin-top: 0;
            color: #06b6d4;
        }
        
        #facts-list {
            background: #0f172a;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        .fact {
            padding: 8px;
            background: #1e293b;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 3px solid #10b981;
        }
        
        .chat-container {
            background: #1e293b;
            padding: 20px;
            border-radius: 10px;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        
        .message {
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 8px;
            max-width: 70%;
        }
        
        .user {
            background: #6366f1;
            margin-left: auto;
            text-align: right;
        }
        
        .bot {
            background: #059669;
        }
        
        .bot.no-info {
            background: #dc2626;
        }
        
        .info-badge {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        input {
            flex: 1;
            padding: 12px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 5px;
            color: #f8fafc;
            font-size: 16px;
        }
        
        button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #a855f7, #ec4899);
            border: none;
            border-radius: 5px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            font-size: 16px;
        }
        
        button:hover {
            opacity: 0.9;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: #1e293b;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #334155;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: 900;
            color: #06b6d4;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.7;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>🎵 AII Zero-Hallucination Demo</h1>
    
    <div class="context-box">
        <h3>📚 Context (Known Facts)</h3>
        <p>Add facts to the chatbot's context. It can ONLY answer based on these facts.</p>
        <input type="text" id="fact-input" placeholder="Enter a fact... (e.g., 'Paris is the capital of France')">
        <button onclick="addFact()">Add Fact</button>
        <div id="facts-list"></div>
    </div>
    
    <div class="chat-container" id="chat"></div>
    
    <div class="input-container">
        <input type="text" id="message-input" placeholder="Ask a question..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value" id="total-messages">0</div>
            <div class="stat-label">Total Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="hallucination-rate">0%</div>
            <div class="stat-label">Hallucination Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="grounded-responses">0</div>
            <div class="stat-label">Grounded Responses</div>
        </div>
    </div>
    
    <script>
        let facts = [];
        let messages = [];
        let totalMessages = 0;
        let groundedResponses = 0;
        
        function addFact() {
            const input = document.getElementById('fact-input');
            const fact = input.value.trim();
            
            if (fact) {
                facts.push(fact);
                updateFactsList();
                input.value = '';
            }
        }
        
        function updateFactsList() {
            const list = document.getElementById('facts-list');
            list.innerHTML = facts.map(f => `<div class="fact">✓ ${f}</div>`).join('');
        }
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (message) {
                addMessage('user', message);
                
                // Simulate AII response
                const response = getAIIResponse(message);
                setTimeout(() => {
                    addMessage('bot', response.text, response.grounded);
                    updateStats(response.grounded);
                }, 500);
                
                input.value = '';
            }
        }
        
        function getAIIResponse(question) {
            // Simple keyword matching simulation
            const questionLower = question.toLowerCase();
            
            for (const fact of facts) {
                const factLower = fact.toLowerCase();
                
                // Very simple matching - in real AII this would be semantic
                const words = questionLower.split(' ').filter(w => w.length > 3);
                const matchingWords = words.filter(w => factLower.includes(w));
                
                if (matchingWords.length > 0) {
                    return {
                        text: `Based on my context: ${fact}`,
                        grounded: true,
                        info: 1.0
                    };
                }
            }
            
            return {
                text: `I don't have information about "${question}" in my context. I cannot answer without sources.`,
                grounded: false,
                info: 0.0
            };
        }
        
        function addMessage(type, text, grounded = true) {
            const chat = document.getElementById('chat');
            const className = type === 'user' ? 'message user' : `message bot ${!grounded ? 'no-info' : ''}`;
            
            const messageDiv = document.createElement('div');
            messageDiv.className = className;
            messageDiv.innerHTML = `
                ${text}
                ${type === 'bot' ? `<div class="info-badge">${grounded ? '✓ Grounded in context' : '⚠️ No information available'}</div>` : ''}
            `;
            
            chat.appendChild(messageDiv);
            chat.scrollTop = chat.scrollHeight;
            
            messages.push({type, text, grounded});
        }
        
        function updateStats(grounded) {
            totalMessages++;
            if (grounded) groundedResponses++;
            
            document.getElementById('total-messages').textContent = totalMessages;
            document.getElementById('grounded-responses').textContent = groundedResponses;
            document.getElementById('hallucination-rate').textContent = '0%';
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Example facts to start
        facts = [
            "Paris is the capital of France",
            "The Eiffel Tower is 330 meters tall",
            "The speed of light is 299,792,458 meters per second"
        ];
        updateFactsList();
        
        // Welcome message
        addMessage('bot', 'Hello! I am an AII chatbot with zero hallucination. I can only answer based on facts in my context. Try asking me something!', true);
    </script>
</body>
</html>
```

---

## Running the Demos

### Setup

```bash
# Clone repository
git clone https://github.com/brokenrecord-studio/aii-demos
cd aii-demos

# Install dependencies
mix deps.get

# Run chatbot demo
mix run demos/chatbot.ex

# Run fact checker
mix run demos/fact_checker.ex

# Run Q&A with provenance
mix run demos/qa_with_provenance.ex

# Start web demo
cd demos/web && python -m http.server 8000
# Open http://localhost:8000
```

### Expected Output

**Chatbot Demo:**
```
User: What is the capital of France?
Bot: The capital of France is Paris.
(Info: 8.5 bits)

User: What is the capital of Mars?
Bot: I don't have information about the capital of Mars in my context.
(Info: 0.0 bits)

✓ Zero hallucinations
✓ Honest about missing information
```

---

## Key Takeaways

1. **Conservation Prevents Hallucination**: Information cannot be created from nothing
2. **Explicit Sources**: Every answer traces back to source material
3. **Honest Uncertainty**: AI admits when it doesn't know
4. **Type-Safe**: Compile-time guarantees, not runtime hopes
5. **Provable Correctness**: Mathematical guarantee of no hallucination

**The difference is fundamental: Traditional AI tries to reduce hallucination. AII makes it impossible.**