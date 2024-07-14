import cv2
import time
import pose_detector as pd


class SemaphoreDecoder:
    def __init__(self, angle_gap=22.5):
        self.angle_gap = angle_gap
        self.angle_ranges = {
            0: (-self.angle_gap, self.angle_gap),
            -45: (-45 - self.angle_gap, -45 + self.angle_gap),
            -90: (-90 - self.angle_gap, -90 + self.angle_gap),
            -135: (-135 - self.angle_gap, -135 + self.angle_gap),
            -180: (-180, -180 + self.angle_gap),
            45: (45 - self.angle_gap, 45 + self.angle_gap),
            90: (90 - self.angle_gap, 90 + self.angle_gap),
            135: (135 - self.angle_gap, 135 + self.angle_gap),
            180: (180 - self.angle_gap, 180),
        }
        self.english_alphabet = {
            (135, 90): 'A',
            (180, 90): 'B',
            (-135, 90): 'C',
            (-90, 90): 'D',
            (90, -45): 'E',
            (90, 0): 'F',
            (90, 45): 'G',
            (180, 135): 'H',
            (-135, 135): 'I',
            (-90, 0): 'J',
            (135, -90): 'K',
            (135, -45): 'L',
            (135, 0): 'M',
            (135, 45): 'N',
            (-135, 180): 'O',
            (180, -90): 'P',
            (180, -45): 'Q',
            (180, 0): 'R',
            (180, 45): 'S',
            (-135, -90): 'T',
            (-135, -45): 'U',
            (-90, 45): 'V',
            (0, -45): 'W',
            (45, -45): 'X',
            (-135, 0): 'Y',
            (45, 0): 'Z',
            (90, -90): 'SPACE',
            (-45, -135): 'STOP'
        }
        self.ukrainian_alphabet = {
            (135, 45): 'А',
            (180, 135): 'Б',
            (180, 90): 'В',
            (90, 0): 'Г/Ґ',
            (45, 0): 'Д',
            (-135, 90): 'E/Є',
            (-135, 0): 'Ж',
            (180, -45): 'З',
            (135, -90): 'И',
            (-90, 90): 'І/Ї/Й',
            (45, -45): 'К',
            (-135, 45): 'Л',
            (135, -45): 'М',
            (135, 90): 'Н',
            (90, 45): 'О',
            (-90, 0): 'П',
            (180, -90): 'Р',
            (90, -45): 'С',
            (180, 0): 'Т',
            (-135, -45): 'У',
            (-90, 45): 'Ф',
            (-135, 135): 'Х',
            (180, 45): 'Ц',
            (135, 0): 'Ч',
            (-90, -45): 'Ш',
            (-135, -90): 'Щ',
            (-90, -90): 'Ь',
            (-135, 180): 'Ю',
            (0, -45): 'Я',
            (90, -90): 'ПРОБІЛ',
            (-45, -135): 'КІНЕЦЬ'
        }

    def match_angle(self, angle):
        for matching_angle, (lower_bound, upper_bound) in self.angle_ranges.items():
            if lower_bound < angle <= upper_bound:
                if matching_angle == -180:
                    matching_angle = 180
                return matching_angle

    def find_letter(self, right_angle, left_angle, language="en"):
        if left_angle is None or right_angle is None:
            return None
        letter_right_angle = self.match_angle(right_angle)
        letter_left_angle = self.match_angle(left_angle)

        if language == "en":
            letter = self.english_alphabet.get((letter_right_angle, letter_left_angle))
        elif language == "uk":
            letter = self.ukrainian_alphabet.get((letter_right_angle, letter_left_angle))
        else:
            letter = None

        return letter


def main():
    cap = cv2.VideoCapture('videos/semaphore_en.mp4')
    detector = pd.PoseDetector()
    decoder = SemaphoreDecoder()
    start_time = 0

    while True:
        success, img = cap.read()
        img, landmarks = detector.find_pose(img)
        if len(landmarks) != 0:
            img, right_angle = detector.find_angle(img, 14, 16)
            img, left_angle = detector.find_angle(img, 13, 15)

            letter = decoder.find_letter(right_angle, left_angle)
            print(letter)

            cv2.rectangle(img, (1110, 450), (1260, 650), (255, 255, 255), cv2.FILLED)
            if letter == 'Numerical sign':
                letter = 'Num'
                cv2.putText(img, str(letter), (1140, 600), cv2.FONT_HERSHEY_PLAIN, 3,
                       (0, 0, 255), 2)
            elif letter == 'Cancel' or letter is None:
                cv2.putText(img, str(letter), (1140, 600), cv2.FONT_HERSHEY_PLAIN, 2,
                           (0, 0, 255), 2)
            else:
                cv2.putText(img, str(letter), (1140, 600), cv2.FONT_HERSHEY_PLAIN, 10,
                           (0, 0, 255), 5)

            stop_time = time.time()
            fps = 1 / (stop_time - start_time)
            start_time = stop_time

            cv2.putText(img, f'FPS: {str(int(fps))}', (20, 50), cv2.FONT_HERSHEY_PLAIN,
                       3,
                       (0, 0, 255), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()