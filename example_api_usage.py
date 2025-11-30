"""
Example script demonstrating how to use the Name Matching API.

This script shows:
1. How to make single predictions
2. How to make batch predictions
3. How to handle responses and errors
4. How to work with different thresholds

Usage:
    # Start the API first
    python app.py

    # In another terminal, run this script
    python example_api_usage.py
"""

import requests


def check_api_health(base_url: str = "http://localhost:5001") -> bool:
    """
    Check if the API is running and healthy.

    :param base_url: Base URL of the API
    :return: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is healthy and running")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Could not connect to API: {e}")
        print("  Make sure the API is running: python app.py")
        return False


def get_model_info(base_url: str = "http://localhost:5001") -> None:
    """
    Get and display model information.

    :param base_url: Base URL of the API
    """
    try:
        response = requests.get(f"{base_url}/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\nModel Information:")
            print(f"  Type: {data['model']['type']}")
            print(f"  Number of features: {data['model']['num_features']}")
            print(f"  Features: {', '.join(data['model']['features'])}")
        else:
            print(f"Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"Error getting model info: {e}")


def example_single_prediction(base_url: str = "http://localhost:5001") -> None:
    """
    Example of making a single prediction.

    :param base_url: Base URL of the API
    """
    print("\n" + "=" * 60)
    print("Example 1: Single Prediction")
    print("=" * 60)

    request_data = {
        "CUST_NAME": "John Smith",
        "COUNTERPART_NAME": "J. Smith",
        "FT_NO": "FT12345",
        "threshold": 0.85,
    }

    print(f"\nRequest:")
    print(f"  Customer Name: {request_data['CUST_NAME']}")
    print(f"  Counterpart Name: {request_data['COUNTERPART_NAME']}")
    print(f"  Transaction: {request_data['FT_NO']}")
    print(f"  Threshold: {request_data['threshold']}")

    try:
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        if response.status_code == 200:
            data = response.json()
            result = data["result"]

            print(f"\nResponse:")
            print(f"  Status: {data['status']}")
            print(f"  Match Label: {result['match_label']}")
            print(f"  Prediction: {result['prediction']}")
            print(
                f"  Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)"
            )
            print(f"  Features:")
            for feature, value in result["features"].items():
                print(f"    - {feature}: {value:.4f}")
        else:
            print(f"\nError: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"Error: {e}")


def example_batch_prediction(base_url: str = "http://localhost:5001") -> None:
    """
    Example of making batch predictions.

    :param base_url: Base URL of the API
    """
    print("\n" + "=" * 60)
    print("Example 2: Batch Prediction")
    print("=" * 60)

    request_data = {
        "pairs": [
            {
                "CUST_NAME": "John Smith",
                "COUNTERPART_NAME": "J. Smith",
                "FT_NO": "FT001",
            },
            {
                "CUST_NAME": "Apple Inc.",
                "COUNTERPART_NAME": "Apple Corporation",
                "FT_NO": "FT002",
            },
            {
                "CUST_NAME": "Microsoft Corporation",
                "COUNTERPART_NAME": "Amazon Web Services",
                "FT_NO": "FT003",
            },
            {
                "CUST_NAME": "Jane Marie Doe",
                "COUNTERPART_NAME": "Jane M. Doe",
                "FT_NO": "FT004",
            },
        ],
        "threshold": 0.85,
    }

    print(f"\nRequest: {len(request_data['pairs'])} name pairs")
    for i, pair in enumerate(request_data["pairs"], 1):
        print(f"  {i}. {pair['CUST_NAME']} <-> {pair['COUNTERPART_NAME']}")

    try:
        response = requests.post(
            f"{base_url}/predict/batch", json=request_data, timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(f"  Status: {data['status']}")
            print(f"  Count: {data['count']}")
            print(f"\nResults:")

            for result in data["results"]:
                print(f"\n  Transaction: {result['ft_no']}")
                print(f"    Names: {result['name_x']} <-> {result['name_y']}")
                print(f"    Match: {result['match_label']}")
                print(
                    f"    Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)"
                )

        else:
            print(f"\nError: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"Error: {e}")


def example_threshold_comparison(base_url: str = "http://localhost:5001") -> None:
    """
    Example showing how different thresholds affect predictions.

    :param base_url: Base URL of the API
    """
    print("\n" + "=" * 60)
    print("Example 3: Threshold Comparison")
    print("=" * 60)

    test_pair = {
        "CUST_NAME": "Jane Doe",
        "COUNTERPART_NAME": "J. Doe",
        "FT_NO": "FT_THRESHOLD_TEST",
    }

    thresholds = [0.5, 0.7, 0.85, 0.95]

    print(
        f"\nTesting with: {test_pair['CUST_NAME']} <-> {test_pair['COUNTERPART_NAME']}"
    )
    print(f"\nThreshold Comparison:")

    try:
        for threshold in thresholds:
            request_data = {**test_pair, "threshold": threshold}
            response = requests.post(
                f"{base_url}/predict", json=request_data, timeout=30
            )

            if response.status_code == 200:
                result = response.json()["result"]
                match_symbol = "✓" if result["prediction"] == 1 else "✗"
                print(
                    f"  {match_symbol} Threshold {threshold:.2f}: "
                    f"{result['match_label']:10s} (prob: {result['probability']:.4f})"
                )
            else:
                print(f"  Error at threshold {threshold}")

    except Exception as e:
        print(f"Error: {e}")


def example_error_handling(base_url: str = "http://localhost:5001") -> None:
    """
    Example showing error handling for invalid inputs.

    :param base_url: Base URL of the API
    """
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    # Test with empty name
    print("\n1. Testing with empty customer name:")
    request_data = {
        "CUST_NAME": "",
        "COUNTERPART_NAME": "John Doe",
        "FT_NO": "FT_ERROR_1",
    }

    try:
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)
        data = response.json()
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {data['status']} - {data['message']}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test with invalid threshold
    print("\n2. Testing with invalid threshold:")
    request_data = {
        "CUST_NAME": "John Doe",
        "COUNTERPART_NAME": "Jane Doe",
        "FT_NO": "FT_ERROR_2",
        "threshold": 1.5,  # Invalid: > 1
    }

    try:
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)
        data = response.json()
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {data['status']} - {data['message']}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test with missing field
    print("\n3. Testing with missing counterpart name:")
    request_data = {"CUST_NAME": "John Doe", "FT_NO": "FT_ERROR_3"}

    try:
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)
        data = response.json()
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {data['status']} - {data['message']}")
    except Exception as e:
        print(f"   Error: {e}")


def main():
    """Run all examples."""
    base_url = "http://localhost:5001"

    print("\n" + "=" * 60)
    print("Name Matching API - Usage Examples")
    print("=" * 60)

    # Check if API is running
    if not check_api_health(base_url):
        return

    # Get model information
    get_model_info(base_url)

    # Run examples
    example_single_prediction(base_url)
    example_batch_prediction(base_url)
    example_threshold_comparison(base_url)
    example_error_handling(base_url)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
