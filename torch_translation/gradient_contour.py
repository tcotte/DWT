import numpy as np
from scipy import ndimage
import cv2

def compute_distance_transform(instance_mask):
    """
    Compute the distance transform from the binary instance mask.
    :param instance_mask: Binary mask of the instance.
    :return: Distance transform of the instance mask.
    """
    distance_transform = ndimage.distance_transform_edt(instance_mask)
    return distance_transform

def compute_normalized_gradient(distance_transform):
    """
    Compute the normalized gradient of the distance transform.
    :param distance_transform: Distance transform.
    :return: Normalized gradient (unit vector field) of the distance transform.
    """
    # Compute gradients
    grad_y, grad_x = np.gradient(distance_transform)

    # Compute the magnitude of the gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Avoid division by zero by setting zero magnitudes to one
    magnitude[magnitude == 0] = 1

    # Normalize the gradients to get the unit vectors
    unit_vector_x = grad_x / magnitude
    unit_vector_y = grad_y / magnitude

    return unit_vector_x, unit_vector_y

def generate_ground_truth_vectors(instance_mask):
    """
    Generate ground truth directional unit vectors from the instance mask.
    :param instance_mask: Binary mask of the instance.
    :return: Two-channel output representing the directional unit vectors.
    """
    # Compute the distance transform
    distance_transform = compute_distance_transform(instance_mask)

    # Compute the normalized gradient of the distance transform
    unit_vector_x, unit_vector_y = compute_normalized_gradient(distance_transform)

    # Stack the unit vectors to form a two-channel output
    directional_unit_vectors = np.stack((unit_vector_x, unit_vector_y), axis=-1)

    return directional_unit_vectors

# Example usage
# Load or create a binary mask for an instance
instance_mask = cv2.imread(r'C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3\masks_machine\2024-05-22-13-39-31-563085.png', cv2.IMREAD_GRAYSCALE)
instance_mask = instance_mask > 0  # Convert to binary

# Generate ground truth directional unit vectors
ground_truth_vectors = generate_ground_truth_vectors(instance_mask)

# Display the result
import matplotlib.pyplot as plt
if __name__ == "__main__":
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Unit Vector X Component')
    plt.imshow(ground_truth_vectors[:, :, 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Unit Vector Y Component')
    plt.imshow(ground_truth_vectors[:, :, 1], cmap='gray')
    plt.show()
