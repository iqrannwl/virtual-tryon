"""
Test script to verify the virtual try-on API setup
"""
import requests
import base64
from PIL import Image
import io


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_models_endpoint():
    """Test the models endpoint"""
    print("\nTesting models endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/models")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_tryon_endpoint(person_image_path: str, garment_image_path: str):
    """Test the virtual try-on endpoint"""
    print("\nTesting virtual try-on endpoint...")
    try:
        with open(person_image_path, 'rb') as person_file, \
             open(garment_image_path, 'rb') as garment_file:
            
            files = {
                'person_image': ('person.jpg', person_file, 'image/jpeg'),
                'garment_image': ('garment.jpg', garment_file, 'image/jpeg')
            }
            
            data = {
                'category': 'upper_body',
                'num_inference_steps': 20,  # Reduced for faster testing
                'guidance_scale': 2.0
            }
            
            print("Sending request...")
            response = requests.post(
                "http://localhost:8000/api/v1/tryon",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result['success']}")
                print(f"Message: {result['message']}")
                print(f"Processing Time: {result['processing_time']} seconds")
                
                # Save the result image
                if 'result_image' in result:
                    img_data = base64.b64decode(result['result_image'])
                    img = Image.open(io.BytesIO(img_data))
                    img.save('test_result.png')
                    print("Result saved to test_result.png")
                
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Virtual Try-On API Test Suite")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_check()
    models_ok = test_models_endpoint()
    
    print("\n" + "=" * 50)
    print("Basic Tests Summary:")
    print(f"Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Models Endpoint: {'✓ PASS' if models_ok else '✗ FAIL'}")
    print("=" * 50)
    
    # Test try-on endpoint (requires sample images)
    print("\nTo test the try-on endpoint, run:")
    print("python test_api.py <person_image_path> <garment_image_path>")
    print("\nExample:")
    print("python test_api.py person.jpg garment.jpg")
