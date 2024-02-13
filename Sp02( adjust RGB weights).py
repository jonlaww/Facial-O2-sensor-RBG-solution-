import cv2
import mediapipe as mp
import numpy as np
import time
import scipy.signal

mp_drawing = mp.solutions.drawing_utils  # Add this line

class SpO2Estimator:
    def __init__(self):
        self.red_values = []
        self.blue_values = []
        self.green_values = []
        self.spo2_values = []

    def estimate_spo2(self, frame_rgb, landmarks):
        red_weight = 0.3
        green_weight = 0.6
        blue_weight = 0.1
        # Define width and height based on landmarks position
        w = int(max(landmarks.landmark[i].x for i in [10, 338, 151, 389, 398, 175, 152, 377]) * frame_rgb.shape[1] - 
            min(landmarks.landmark[i].x for i in [10, 338, 151, 389, 398, 175, 152, 377]) * frame_rgb.shape[1])
        h = int(max(landmarks.landmark[i].y for i in [10, 338, 151, 389, 398, 175, 152, 377]) * frame_rgb.shape[0] - 
            min(landmarks.landmark[i].y for i in [10, 338, 151, 389, 398, 175, 152, 377]) * frame_rgb.shape[0])

        # Get forehead region
        forehead_center_x = int((landmarks.landmark[10].x + landmarks.landmark[338].x + landmarks.landmark[151].x +
                            landmarks.landmark[389].x + landmarks.landmark[398].x + landmarks.landmark[175].x +
                            landmarks.landmark[152].x + landmarks.landmark[377].x) / 8 * frame_rgb.shape[1])
        forehead_center_y = int((landmarks.landmark[10].y + landmarks.landmark[338].y + landmarks.landmark[151].y +
                            landmarks.landmark[389].y + landmarks.landmark[398].y + landmarks.landmark[175].y +
                            landmarks.landmark[152].y + landmarks.landmark[377].y) / 8 * frame_rgb.shape[0])
        forehead_roi = frame_rgb[forehead_center_y - h // 8:forehead_center_y + h // 8,
                            forehead_center_x - w // 8:forehead_center_x + w // 8]

        # Get left and right cheek regions
        left_cheek_x = int((landmarks.landmark[4].x + landmarks.landmark[1].x) / 2 * frame_rgb.shape[1])
        left_cheek_y = int((landmarks.landmark[4].y + landmarks.landmark[1].y) / 2 * frame_rgb.shape[0])
        right_cheek_x = int((landmarks.landmark[15].x + landmarks.landmark[12].x) / 2 * frame_rgb.shape[1])
        right_cheek_y = int((landmarks.landmark[15].y + landmarks.landmark[12].y) / 2 * frame_rgb.shape[0])

        # Extract left and right cheek ROIs
        left_cheek_roi = frame_rgb[left_cheek_y - h // 8:left_cheek_y + h // 8, left_cheek_x - w // 8:left_cheek_x + w // 8]
        right_cheek_roi = frame_rgb[right_cheek_y - h // 8:right_cheek_y + h // 8, right_cheek_x - w // 8:right_cheek_x + w // 8]

        # Average color from forehead, left cheek, and right cheek ROIs
        avg_color_forehead = cv2.mean(forehead_roi)[:3]
        avg_color_left_cheek = cv2.mean(left_cheek_roi)[:3]
        avg_color_right_cheek = cv2.mean(right_cheek_roi)[:3]

        # Average color from forehead, left cheek, and right cheek
        avg_color = [(avg_color_forehead[i] + avg_color_left_cheek[i] + avg_color_right_cheek[i]) / 3 for i in range(3)]



        # Add red and blue values
        self.red_values.append(avg_color[2])  # Use avg_color instead of avg_color_forehead
        self.blue_values.append(avg_color[0]) # Use avg_color instead of avg_color_forehead
        self.green_values.append(avg_color[1])  # Adding green channel


        if len(self.red_values) > 100:
            self.red_values.pop(0)
            self.blue_values.pop(0)
            self.green_values.pop(0)

        if len(self.red_values) == 100:
            # Apply a low-pass filter to the red and blue values
            b, a = scipy.signal.butter(4, 0.15)
            filtered_red_values = scipy.signal.lfilter(b, a, self.red_values)
            filtered_blue_values = scipy.signal.lfilter(b, a, self.blue_values)
            filtered_green_values = scipy.signal.lfilter(b, a, self.green_values)

            # Compute the mean of the filtered values
            mean_filtered_green_value = np.mean(filtered_green_values)
            mean_filtered_red_value = np.mean(filtered_red_values)
            mean_filtered_blue_value = np.mean(filtered_blue_values)
            mean_filtered_total_value = (mean_filtered_red_value + mean_filtered_blue_value + mean_filtered_green_value) / 3


            # Compute SpO2 based on the filtered values
            spo2_value = min(100, 100 - 5 * (mean_filtered_red_value / mean_filtered_total_value))
            self.spo2_values.append((time.time(), spo2_value))

            # Average the last 10 seconds of SpO2 values
            current_time = time.time()
            self.spo2_values = [(t, v) for t, v in self.spo2_values if current_time - t <= 10]
            avg_spo2 = np.mean([v for t, v in self.spo2_values])

            return round(avg_spo2, 2)
        
        return None

def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2)
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return

    spo2_estimator = SpO2Estimator()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                spo2_value = spo2_estimator.estimate_spo2(image_rgb, face_landmarks)
                mp_drawing.draw_landmarks(image, face_landmarks)

        # Remove SpO2 values older than 10 seconds
        current_time = time.time()
        spo2_estimator.spo2_values = [(t, v) for t, v in spo2_estimator.spo2_values if current_time - t <= 10]

        # Compute average of SpO2 values in the last 10 seconds
        if spo2_estimator.spo2_values:
            avg_spo2 = np.mean([v for t, v in spo2_estimator.spo2_values])
            cv2.putText(image, f"Average SpO2 (last 10s): {avg_spo2:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow('SpO2 Estimation', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



