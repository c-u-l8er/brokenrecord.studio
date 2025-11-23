defmodule SimpleBench do
  def run do
    Benchee.run(%{
      "Simple Test" => fn -> :timer.sleep(1) end
    })
  end
end

SimpleBench.run()
