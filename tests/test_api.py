import requests
import time

API_URL = "https://zaykats-youtube-sentiment-api.hf.space"

# ---------------------------------------------------
# TEST 1 — HEALTH CHECK
# ---------------------------------------------------
def test_health():
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    print(" Test health passed")


# ---------------------------------------------------
# TEST 2 — PREDICT NORMAL CASE
# ---------------------------------------------------
def test_predict_basic():
    data = {
        "comments": [
            "This is amazing!",
            "I don't like this",
            "It's okay"
        ]
    }
    response = requests.post(f"{API_URL}/predict_batch", json=data)
    assert response.status_code == 200

    result = response.json()
    assert len(result["predictions"]) == 3

    print(" Test predict_basic passed")


# ---------------------------------------------------
# TEST 3 — BATCH SIZE TESTS
# ---------------------------------------------------
def test_batch_sizes():
    batches = [
        ["ok"],
        ["good", "bad", "neutral"],
        ["test"] * 50,
        ["test"] * 200
    ]

    for batch in batches:
        response = requests.post(f"{API_URL}/predict_batch", json={"comments": batch})
        assert response.status_code == 200
        print(f" Test batch size {len(batch)} passed")


# ---------------------------------------------------
# TEST 4 — ERROR HANDLING
# ---------------------------------------------------
def test_error_handling():
    invalid_inputs = [
        {"comments": []},               # empty batch
        {},                              # missing key
        {"comments": "not a list"},     # wrong type
        {"comments": [123, None]},      # invalid elements
    ]

    for case in invalid_inputs:
        response = requests.post(f"{API_URL}/predict_batch", json=case)
        print(f"Case {case} → HTTP {response.status_code}")
        assert response.status_code in [400, 422]
    
    print(" Test error handling passed")


# ---------------------------------------------------
# TEST 5 — LATENCY (TIME OF RESPONSE)
# ---------------------------------------------------
def test_api_latency():
    start = time.time()
    response = requests.post(f"{API_URL}/predict_batch", json={"comments": ["Great video!"]})
    end = time.time()

    assert response.status_code == 200

    latency_ms = (end - start) * 1000
    print(f" API latency: {latency_ms:.2f} ms")
    print(" Test API latency passed")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n Running full test suite...\n")

    test_health()
    test_predict_basic()
    test_batch_sizes()
    test_error_handling()
    test_api_latency()

    print("\n ALL TESTS PASSED SUCCESSFULLY! ")
