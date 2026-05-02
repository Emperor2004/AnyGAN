# location: /tests/test_base_model.py

import pytest

from models.base_model import BaseImageModel


def test_base_model_generate_contract_raises():
    model = BaseImageModel(
        display_name="Base",
        slug="base",
        model_type="abstract",
        description="abstract base",
    )

    with pytest.raises(NotImplementedError, match="generate"):
        model.generate({})
