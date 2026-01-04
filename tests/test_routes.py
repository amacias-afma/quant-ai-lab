import pytest

def test_home_page(client):
    """
    Test that the Landing Page (Portfolio) loads correctly.
    """
    response = client.get('/')
    assert response.status_code == 200
    # Check if the title from index.html is present
    assert b"Quant AI Lab" in response.data

def test_volatility_dashboard_route(client):
    """
    Test that the Project Dashboard loads correctly.
    """
    response = client.get('/projects/volatility')
    assert response.status_code == 200
    # Check for specific content from project_volatility.html
    # (Note: Ensure this string exists in your template, e.g., in the Navbar)
    assert b"Basel III" in response.data

def test_404_route(client):
    """
    Test that a non-existent route returns a 404 error.
    """
    response = client.get('/non-existent-page')
    assert response.status_code == 404