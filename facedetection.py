import cv2
import numpy as np
import os
# 1. Haar-like features
def calculate_integral_image(image):
    integral_image = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return integral_image

def apply_haar_feature(integral_image, feature_type, top_left, bottom_right):
    top_left_value = integral_image[top_left[0], top_left[1]]
    top_right_value = integral_image[top_left[0], bottom_right[1]]
    bottom_left_value = integral_image[bottom_right[0], top_left[1]]
    bottom_right_value = integral_image[bottom_right[0], bottom_right[1]]

    if feature_type == "edge":
        return top_left_value + bottom_right_value - (top_right_value + bottom_left_value)
    elif feature_type == "line":
        return top_right_value + bottom_left_value - (top_left_value + bottom_right_value)
    elif feature_type == "four-sided":
        return top_left_value + bottom_right_value - (top_right_value + bottom_left_value)

# 2. Integral image
def integral_image_from_gray_image(gray_image):
    integral_image = calculate_integral_image(gray_image)
    return integral_image

# 3. Ada-boost
def train_weak_classifier(integral_images, labels, feature_type, top_left, bottom_right):
    features = []

    for integral_image in integral_images:
        feature_value = apply_haar_feature(integral_image, feature_type, top_left, bottom_right)
        features.append(feature_value)

    features = np.array(features)
    error = np.sum((features - labels) ** 2)

    return error, feature_type, top_left, bottom_right

def ada_boost(integral_images, labels, num_classifiers):
    classifiers = []

    for _ in range(num_classifiers):
        # Select a random feature type and rectangle
        feature_type = np.random.choice(["edge", "line", "four-sided"])
        top_left = np.random.randint(0, integral_images[0].shape[0], size=2)
        bottom_right = np.random.randint(top_left[0], integral_images[0].shape[0], size=2)

        # Train a weak classifier
        error, feature_type, top_left, bottom_right = train_weak_classifier(
            integral_images, labels, feature_type, top_left, bottom_right
        )

        # Calculate the classifier weight
        classifier_weight = 0.5 * np.log((1 - error) / error)

        # Save the classifier
        classifiers.append((classifier_weight, feature_type, top_left, bottom_right))

    return classifiers

# 4. Cascading
def apply_cascade_classifiers(images, classifiers, scale_factor=1.3, min_neighbors=5):
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = []

        for scale in [scale_factor ** i for i in range(5)]:
            resized_image = cv2.resize(gray_image, (int(gray_image.shape[1] / scale), int(gray_image.shape[0] / scale)))

            integral_image = integral_image_from_gray_image(resized_image)

            for classifier in classifiers:
                classifier_weight, feature_type, top_left, bottom_right = classifier

                # Apply the weak classifier
                feature_value = apply_haar_feature(integral_image, feature_type, top_left, bottom_right)

                if feature_value * classifier_weight < 0:
                    break
            else:
                # All weak classifiers passed, consider it a face
                faces.append(
                    (
                        int(top_left[0] * scale),
                        int(top_left[1] * scale),
                        int((bottom_right[0] - top_left[0]) * scale),
                        int((bottom_right[1] - top_left[1]) * scale),
                    )
                )

    return faces

images = []

for roots,dirs,files in os.walk("/mnt/c/Users/kunda/Downloads/faces94/"):
    
    for file in files:
        img = cv2.imread(os.path.join(roots,file))
        if img is not None:
            images.append(img)


# # Load the image
# image = cv2.imread("path/to/your/image.jpg")

# Mock labels (assuming positive examples)
labels = np.random.choice([-1, 1], size=len(images))

# Mock integral images
integral_images = [integral_image_from_gray_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]

# Train Ada-boost classifiers
classifiers = ada_boost(integral_images, labels, num_classifiers=10)

# Apply cascading classifiers to detect faces
detected_faces = apply_cascade_classifiers(images, classifiers)


for face in detected_faces:
    x, y, w, h = face
    cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
