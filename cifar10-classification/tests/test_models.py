import pytest
import torch
from torch import nn

from cifar10.models import CifarCNN, PlainBlock, ResidualBlock, build_model
from cifar10.utils import count_parameters, set_seed


@pytest.mark.parametrize("name", ["baseline", "residual"])
def test_output_shape(name):
    model = build_model(name)
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)


@pytest.mark.parametrize("name", ["baseline", "residual"])
def test_has_required_components(name):
    modules = list(build_model(name).modules())
    assert any(isinstance(m, nn.Conv2d) for m in modules)
    assert any(isinstance(m, nn.BatchNorm2d) for m in modules)
    assert any(isinstance(m, nn.Dropout) for m in modules)
    assert isinstance(build_model(name).head[-1], nn.Linear)


def test_residual_block_shapes_and_shortcut():
    same = ResidualBlock(16, 16)
    assert isinstance(same.shortcut, nn.Identity)
    assert same(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)

    projected = ResidualBlock(16, 32)
    assert isinstance(projected.shortcut, nn.Sequential)
    assert projected(torch.randn(2, 16, 8, 8)).shape == (2, 32, 8, 8)


def test_residual_block_adds_input():
    """With the conv branch zeroed out, the block reduces to ReLU(x)."""
    block = ResidualBlock(8, 8).eval()
    nn.init.zeros_(block.bn2.weight)
    nn.init.zeros_(block.bn2.bias)
    x = torch.randn(2, 8, 4, 4)
    assert torch.allclose(block(x), torch.relu(x))


def test_models_differ_only_by_block():
    baseline, residual = build_model("baseline"), build_model("residual")
    assert all(isinstance(m, PlainBlock) for m in baseline.features if isinstance(m, (PlainBlock, ResidualBlock)))
    assert sum(isinstance(m, ResidualBlock) for m in residual.features) == 3
    # The only extra parameters are the 1x1 projection shortcuts: within 5%.
    ratio = count_parameters(residual) / count_parameters(baseline)
    assert 1.0 < ratio < 1.05


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("vgg")


def test_gradients_flow_to_first_layer():
    model = build_model("residual")
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 32, 32)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    assert first_conv.weight.grad is not None and first_conv.weight.grad.abs().sum() > 0


def test_seed_gives_identical_initialization():
    set_seed(0)
    a = CifarCNN(PlainBlock)
    set_seed(0)
    b = CifarCNN(PlainBlock)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
