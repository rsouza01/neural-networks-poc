from neural_networks_poc import NeuralNetwork


def test_forward_pass_returns_expected_length() -> None:
    network = NeuralNetwork(layers=[4, 3, 2])
    result = network.forward([1.0, 2.0, 3.0, 4.0])
    assert len(result) == 2


def test_forward_pass_raises_on_bad_input_length() -> None:
    network = NeuralNetwork(layers=[4, 2])
    try:
        network.forward([1.0, 2.0, 3.0])
    except ValueError as exc:
        assert "Input length must match" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid input length")
