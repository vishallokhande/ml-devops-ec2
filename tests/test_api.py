# tests/test_api.py
import requests

def test_health():
    r = requests.get('http://65.0.94.40:5000/health')
    assert r.status_code == 200
