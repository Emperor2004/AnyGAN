# location: /tests/test_base_model.py

import pytest

from models.base_model import BaseGenerativeModel


def test_base_model_generate_contract_raises():
    model = BaseGenerativeModel(
        display_name="Base",
        slug="base",
        model_type="abstract",
        description="abstract base",
    )

    with pytest.raises(NotImplementedError, match="generate"):
        model.generate({})


def test_base_model_checkpoint_hook_raises(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")

    with pytest.raises(NotImplementedError, match="Checkpoint loading"):
        BaseGenerativeModel(
            display_name="Base",
            slug="base",
            model_type="abstract",
            description="abstract base",
            checkpoint_path=str(checkpoint),
        )
