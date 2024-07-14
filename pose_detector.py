import time
import math
import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmarks = []

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode,
                                      self.model_complexity,
                                      self.smooth_landmarks,
                                      self.enable_segmentation,
                                      self.smooth_segmentation,
                                      self.min_detection_confidence,
                                      self.min_tracking_confidence)
        self.results = None
        self.landmarks = []

    def find_pose(self, img, visibility=0.5, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        self.landmarks = []

        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                if (0 <= landmark.x <= 1) and (0 <= landmark.y <= 1) and (landmark.visibility > visibility):
                    confident = True
                else:
                    confident = False

                height, weight, _ = img.shape
                x, y = int(landmark.x * weight), int(landmark.y * height)
                self.landmarks.append([idx, x, y, confident])

            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(color=(165, 106, 31), thickness=2, circle_radius=2),
                                            self.mp_draw.DrawingSpec(color=(165, 106, 31), thickness=2, circle_radius=2))

        return img, self.landmarks

    def find_angle(self, img, point1, point2, draw=True):
        x1, y1, c1 = self.landmarks[point1][1:]
        x2, y2, c2 = self.landmarks[point2][1:]
        angle_degrees = None

        if c1 and c2:
            angle_radians = math.atan2(y2 - y1, x2 - x1)
            angle_degrees = round(math.degrees(angle_radians), 1)

        if draw:
            cv2.circle(img, (x1, y1), 5, (165, 106, 31), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (165, 106, 31), 2)
            cv2.circle(img, (x2, y2), 5, (165, 106, 31), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (165, 106, 31), 2)
            if c1 and c2:
                cv2.putText(img, str(angle_degrees), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img, angle_degrees


def main():
    cap = cv2.VideoCapture('videos/semaphore_en.mp4')
    detector = PoseDetector()
    start_time = 0

    while True:
        success, img = cap.read()
        img, landmarks = detector.find_pose(img)

        stop_time = time.time()
        fps = 1 / (stop_time - start_time)
        start_time = stop_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
