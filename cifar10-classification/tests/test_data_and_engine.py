import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10.data import build_transforms, denormalize, split_indices
from cifar10.engine import evaluate, train_one_epoch
from cifar10.models import build_model
from cifar10.utils import set_seed


def test_split_is_disjoint_complete_and_deterministic():
    train, val = split_indices(1000, 100, seed=1)
    assert len(train) == 900 and len(val) == 100
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(1000))
    assert split_indices(1000, 100, seed=1) == (train, val)
    assert split_indices(1000, 100, seed=2) != (train, val)


def test_transforms_output_normalized_tensor():
    image = Image.new("RGB", (32, 32), color=(128, 64, 200))
    for augment in (True, False):
        tensor = build_transforms(augment)(image)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32


def test_denormalize_inverts_normalization():
    image = Image.new("RGB", (32, 32), color=(255, 0, 128))
    restored = denormalize(build_transforms(False)(image))
    expected = torch.tensor([1.0, 0.0, 128 / 255]).view(3, 1, 1).expand(3, 32, 32)
    assert torch.allclose(restored, expected, atol=1e-5)


def test_training_loop_can_overfit_a_tiny_batch():
    """Sanity check of the full loop: the model should memorize 32 random images."""
    set_seed(0)
    images, labels = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=32)
    model = build_model("residual", dropout=0.0, head_dropout=0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    first_loss, _ = train_one_epoch(model, loader, criterion, optimizer, device)
    for _ in range(40):
        train_one_epoch(model, loader, criterion, optimizer, device)
    loss, acc = evaluate(model, loader, criterion, device)
    assert loss < first_loss
    assert acc == 1.0
