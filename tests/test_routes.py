import pytest
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Verificar que la página de inicio carga correctamente."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Quant AI Lab" in rv.data

def test_project_route(client):
    """Verificar que el dashboard de volatilidad carga."""
    rv = client.get('/projects/volatility')
    assert rv.status_code == 200
    assert b"Hybrid Risk Engine" in rv.data