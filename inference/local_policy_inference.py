import numpy as np
import torch
import cv2
import time
import argparse
import logging
import random
from typing import Dict, List, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockPolicyModel:
    """
    A mock model for testing inference with random inputs
    Can be replaced with a real model loading and inference
    """
    def __init__(self, 
                 device: str = "cpu", 
                 model_path: str = None,
                 input_image_size: Tuple[int, int] = (480, 640)):
        """Initialize the mock model"""
        self.device = device
        self.model_path = model_path or "mock_model"
        self.input_image_size = input_image_size
        self.state_size = 7
        self.output_size = 7  # 6 joints + 1 gripper
        
        logger.info(f"Initialized model on {device}")
        logger.info(f"Input image size: {input_image_size}")
        logger.info(f"State size: {self.state_size}")
        logger.info(f"Output size: {self.output_size}")
    
    def __call__(self, 
                 wrist_image: torch.Tensor, 
                 head_image: Union[torch.Tensor, None], 
                 state: torch.Tensor) -> torch.Tensor:
        """
        Mock inference function
        
        Args:
            wrist_image: Wrist camera image tensor [C, H, W]
            head_image: Head camera image tensor [C, H, W] or None
            state: State tensor [state_size]
            
        Returns:
            Prediction tensor [output_size]
        """
        # Simulated inference delay
        time.sleep(0.02)  # 20ms delay
        
        # In a real model, you'd do inference here
        # For mock, we just return random values close to the state
        # This simulates a policy that tries to maintain position with small changes
        base_output = state[:6]  # First 6 values from state (joint positions)
        
        # Add small random offsets to simulate a policy output
        joint_noise = torch.randn(6) * 2.0  # Small random changes
        gripper_value = torch.tensor([random.uniform(-165.0, 0.0)])  # Random gripper angle
        
        # Combine joints and gripper
        prediction = torch.cat([base_output + joint_noise, gripper_value])
        
        return prediction

def generate_random_images(
        wrist_image_size: Tuple[int, int] = (480, 640),
        head_image_size: Tuple[int, int] = (480, 640)
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random images for testing
    
    Args:
        wrist_image_size: Size of wrist camera image (H, W)
        head_image_size: Size of head camera image (H, W)
        
    Returns:
        Tuple of (wrist_image, head_image) as numpy arrays
    """
    # Create random wrist image
    wrist_h, wrist_w = wrist_image_size
    wrist_image = np.random.random((wrist_h, wrist_w, 3)).astype(np.float32)
    
    # Create random head image
    head_h, head_w = head_image_size
    head_image = np.random.random((head_h, head_w, 3)).astype(np.float32)
    
    return wrist_image, head_image

def generate_random_state(size: int = 7) -> List[float]:
    """
    Generate random state vector
    
    Args:
        size: Size of state vector
        
    Returns:
        List of floats representing the state
    """
    # Random joint angles (first 6 values)
    joints = [random.uniform(-180.0, 180.0) for _ in range(6)]
    
    # Random gripper angle (last value)
    gripper = random.uniform(-165.0, 0.0)
    
    return joints + [gripper]

def preprocess_images(
        wrist_image: np.ndarray, 
        head_image: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess images for model input
    
    Args:
        wrist_image: Wrist camera image as numpy array (H, W, C)
        head_image: Head camera image as numpy array (H, W, C)
        
    Returns:
        Tuple of processed images as torch tensors (C, H, W)
    """
    # Convert numpy arrays to torch tensors and permute dimensions
    wrist_tensor = torch.from_numpy(wrist_image).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
    head_tensor = torch.from_numpy(head_image).permute(2, 0, 1)    # [H,W,C] -> [C,H,W]
    
    return wrist_tensor, head_tensor

def local_inference(
        model: Any,
        wrist_image: np.ndarray,
        head_image: np.ndarray,
        state: List[float]
    ) -> Tuple[List[float], float]:
    """
    Run local inference with the model
    
    Args:
        model: Policy model
        wrist_image: Wrist camera image as numpy array (H, W, C)
        head_image: Head camera image as numpy array (H, W, C)
        state: State vector as list of floats
        
    Returns:
        Tuple of (prediction, inference_time_ms)
    """
    # Preprocess images
    wrist_tensor, head_tensor = preprocess_images(wrist_image, head_image)
    
    # Convert state to tensor
    state_tensor = torch.tensor(state, dtype=torch.float32)
    
    # Time the inference
    start_time = time.perf_counter()
    
    # Run inference
    prediction = model(wrist_tensor, head_tensor, state_tensor)
    
    # Calculate inference time
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
    
    return prediction.tolist(), inference_time_ms

def main():
    parser = argparse.ArgumentParser(description="Local Policy Inference")
    parser.add_argument("--device", default="mps", help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--model", default="/Users/jack/Desktop/dummy_ctrl/checkpoints/diff_cube_0406/080000/pretrained_model", help="Path to model (optional)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    parser.add_argument("--wrist_image_size", default="720x1280", help="Wrist camera image size (HxW)")
    parser.add_argument("--head_image_size", default="720x1280", help="Head camera image size (HxW)")
    parser.add_argument("--show_images", action="store_true", help="Display generated images")
    args = parser.parse_args()
    
    # Parse image sizes
    wrist_h, wrist_w = map(int, args.wrist_image_size.split('x'))
    head_h, head_w = map(int, args.head_image_size.split('x'))
    
    # Initialize model
    model = MockPolicyModel(
        device=args.device,
        model_path=args.model,
        input_image_size=(wrist_h, wrist_w)
    )
    
    # Statistics tracking
    total_inference_time = 0.0
    inference_count = 0
    
    try:
        logger.info(f"Running {args.iterations} inference iterations...")
        
        for i in range(args.iterations):
            # Generate random inputs
            wrist_image, head_image = generate_random_images(
                wrist_image_size=(wrist_h, wrist_w),
                head_image_size=(head_h, head_w)
            )
            state = generate_random_state()
            
            # Show images if requested
            if args.show_images and i == 0:  # Show only first iteration
                # Convert to uint8 for display
                wrist_display = (wrist_image * 255).astype(np.uint8)
                head_display = (head_image * 255).astype(np.uint8)
                
                # Display images
                cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_display, cv2.COLOR_RGB2BGR))
                cv2.imshow("Head Camera", cv2.cvtColor(head_display, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1000)  # Wait 1 second
            
            # Run inference
            prediction, inference_time_ms = local_inference(model, wrist_image, head_image, state)
            
            # Update statistics
            total_inference_time += inference_time_ms
            inference_count += 1
            
            # Display results
            logger.info(f"Iteration {i + 1}/{args.iterations} - Inference time: {inference_time_ms:.2f}ms")
            logger.info(f"Input state: {[round(x, 2) for x in state]}")
            logger.info(f"Prediction: {[round(x, 2) for x in prediction]}")
            logger.info(f"Delta: {[round(prediction[j] - state[j], 2) for j in range(len(state))]}")
            logger.info("---")
            
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if args.show_images:
            cv2.destroyAllWindows()
        
        # Display summary statistics
        if inference_count > 0:
            logger.info("\n--- Performance Summary ---")
            logger.info(f"Completed {inference_count} iterations")
            logger.info(f"Average inference time: {total_inference_time / inference_count:.2f}ms")
            logger.info(f"Total processing time: {total_inference_time / 1000:.2f}s")

if __name__ == "__main__":
    main() 