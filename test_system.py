#!/usr/bin/env python3
"""Test script to verify the full system works"""

import requests
import json
import sys

def test_api():
    """Test all API endpoints"""
    print("="*70)
    print("PREDICTIVE MAINTENANCE AI - SYSTEM VERIFICATION")
    print("="*70)
    print()
    
    # Test 1: Health check
    print("1. Testing /health endpoint...")
    try:
        resp = requests.get("http://localhost:8003/health", timeout=5)
        print(f"   ✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   ✅ Response: {data}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    print()
    
    # Test 2: Failure prediction
    print("2. Testing /predict/failure endpoint...")
    features = [1.0] * 364  # 364 features required
    payload = {"features": features}
    try:
        resp = requests.post(
            "http://localhost:8003/predict/failure",
            json=payload,
            timeout=10
        )
        print(f"   ✅ Status: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"   ✅ Failure Probability: {result['failure_probability']:.4f}")
            print(f"   ✅ Failure Imminent: {result['failure_imminent']}")
        else:
            print(f"   ❌ Error: {resp.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    print()
    
    # Test 3: RUL prediction
    print("3. Testing /predict/rul endpoint...")
    try:
        resp = requests.post(
            "http://localhost:8003/predict/rul",
            json=payload,
            timeout=10
        )
        print(f"   ✅ Status: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"   ✅ Estimated RUL: {result['estimated_rul']:.2f} cycles")
        else:
            print(f"   ❌ Error: {resp.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    print()
    
    print("="*70)
    print("✅ ALL TESTS PASSED - SYSTEM IS FULLY OPERATIONAL")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
