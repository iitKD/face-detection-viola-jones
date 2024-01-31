import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

class HaarFeature:
    def __init__(self, feature_type, position, size):
        self.feature_type = feature_type
        self.position = position
        self.size = size
        self.weight = 0.0

def integral_image(image):
    integral_img = np.zeros_like(image, dtype=np.uint32)
    integral_img[0, 0] = image[0, 0]

    for i in range(1, image.shape[0]):
        integral_img[i, 0] = integral_img[i - 1, 0] + image[i, 0]

    for j in range(1, image.shape[1]):
        integral_img[0, j] = integral_img[0, j - 1] + image[0, j]

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            integral_img[i, j] = (
                image[i, j]
                + integral_img[i - 1, j]
                + integral_img[i, j - 1]
                - integral_img[i - 1, j - 1]
            )

    return integral_img

def apply_classifier(integral_img, feature):
    pos = feature.position
    size = feature.size

    if feature.feature_type == "edge":
        return (
            integral_img[pos[0] + size[0], pos[1]]
            - integral_img[pos[0], pos[1]]
            - integral_img[pos[0] + size[0], pos[1] + size[1]]
            + integral_img[pos[0], pos[1] + size[1]]
        )

    elif feature.feature_type == "line":
        return (
            integral_img[pos[0] + size[0], pos[1]]
            - 2 * integral_img[pos[0], pos[1]]
            + integral_img[pos[0] + size[0], pos[1] + size[1]]
        )

    elif feature.feature_type == "four-sided":
        return (
            integral_img[pos[0], pos[1]]
            + integral_img[pos[0] + size[0], pos[1] + size[1]]
            - integral_img[pos[0], pos[1] + size[1]]
            - integral_img[pos[0] + size[0], pos[1]]
        )

def adaboost(features, positive_samples, negative_samples, num_classifiers):
    num_positives = len(positive_samples)
    num_negatives = len(negative_samples)

    weights = np.ones(num_positives + num_negatives) / (num_positives + num_negatives)
    classifiers = []

    for _ in range(num_classifiers):
        best_error = float("inf")
        best_classifier = None

        for feature in features:
            errors = []
            for i, sample in enumerate(positive_samples + negative_samples):
                predicted = 1 if apply_classifier(sample, feature) < feature.threshold else -1
                errors.append(weights[i] * (predicted != 1))

            total_error = sum(errors)
            if total_error < best_error:
                best_error = total_error
                best_classifier = feature

        best_classifier.weight = 0.5 * np.log((1 - best_error) / best_error)
        classifiers.append(best_classifier)

        for i, sample in enumerate(positive_samples + negative_samples):
            predicted = 1 if apply_classifier(sample, best_classifier) < best_classifier.threshold else -1
            weights[i] *= np.exp(-best_classifier.weight * predicted * 1)

        weights /= np.sum(weights)

    return classifiers

def detect_faces(image, classifiers, scale_factor=1.3, min_neighbors=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = []

    for scale in range(1, 5):  # Change the range based on the image size
        resized_image = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale))
        integral_img = integral_image(resized_image)

        for i in range(0, resized_image.shape[0] - 24, 4):
            for j in range(0, resized_image.shape[1] - 24, 4):
                region = integral_img[i : i + 24, j : j + 24]

                for classifier in classifiers:
                    if apply_classifier(region, classifier) < classifier.weight:
                        x, y, w, h = j * scale, i * scale, 24 * scale, 24 * scale
                        faces.append((x, y, w, h))

    # Post-process to remove overlapping rectangles
    faces = cv2.groupRectangles(faces, min_neighbors, 0.2)[0]

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

# Example usage:
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

# Define Haar-like features
features = [
    HaarFeature("edge", (0, 0), (12, 24)),
    HaarFeature("line", (0, 0), (24, 12)),
    HaarFeature("four-sided", (0, 0), (12, 12)),
]

# Generate positive and negative samples (you need a labeled dataset)
positive_samples = []
negative_samples = []

# Implement loading positive and negative samples from your dataset

# Train Adaboost
classifiers = adaboost(features, positive_samples, negative_samples, num_classifiers=5)

# Detect faces
result_image = detect_faces(image.copy(), classifiers)

# Display the result
cv2.imshow("Face Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
