defmodule BrokenRecordZero.MixProject do
  use Mix.Project

  def project do
    [
      app: :broken_record_zero,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix compile" to compile the C NIF
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # No dependencies needed for standalone version
    ]
  end
end
