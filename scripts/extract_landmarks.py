import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands

def extract_landmarks(image):
    landmarks = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    return landmarks if len(landmarks) == 63 else None


def process_dataset(input_folder, output_csv):
    header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Loop through each label folder (A, B, C...)
        for label in os.listdir(input_folder):
            label_folder = os.path.join(input_folder, label)

            if not os.path.isdir(label_folder):  
                continue

            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)

                image = cv2.imread(img_path)
                if image is None:
                    continue

                landmarks = extract_landmarks(image)

                if landmarks:
                    writer.writerow([label] + landmarks)

    print("Landmarks saved to:", output_csv)


if __name__ == "__main__":
    input_path = "dataset/processed"
    output_path = "dataset/landmarks/landmarks.csv"
    process_dataset(input_path, output_path)
