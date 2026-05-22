"""Тесты качества модели линейной регрессии на разных датасетах."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

R2_THRESHOLD = 0.90


def make_linear_dataset(slope=2.0, intercept=1.0, noise_std=0.5, n=100, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, n).reshape(-1, 1)
    y = slope * X.ravel() + intercept + rng.normal(0, noise_std, n)
    return X, y


@pytest.fixture(scope="module")
def trained_model():
    """Модель, обученная на первом чистом датасете."""
    X, y = make_linear_dataset(seed=1)
    return LinearRegression().fit(X, y)


@pytest.mark.parametrize(
    "name,seed",
    [
        ("clean_dataset_1", 1),
        ("clean_dataset_2", 2),
        ("clean_dataset_3", 3),
    ],
)
def test_model_on_clean_data(trained_model, name, seed):
    """Модель должна иметь R² >= 0.90 на чистых данных."""
    X, y = make_linear_dataset(seed=seed)
    r2 = r2_score(y, trained_model.predict(X))
    assert r2 >= R2_THRESHOLD, (
        f"R² на {name} = {r2:.3f}, ожидалось >= {R2_THRESHOLD}"
    )


@pytest.mark.xfail(
    reason="Ожидаемое падение: модель деградирует на зашумлённых данных (σ=5.0)"
)
def test_model_on_noisy_data(trained_model):
    """Тест выявляет проблему: R² ниже порога на зашумлённых данных."""
    X, y = make_linear_dataset(noise_std=5.0, seed=99)
    r2 = r2_score(y, trained_model.predict(X))
    assert r2 >= R2_THRESHOLD, (
        f"ПРОБЛЕМА ОБНАРУЖЕНА: R² на зашумлённых данных = {r2:.3f} "
        f"ниже порога {R2_THRESHOLD}. Модель не справляется с шумом."
    )
