import cv2
from deepface import DeepFace

def analyze_face(image_path):
    print(f"Loading DeepFace model and analyzing '{image_path}'...")
    
    try:
        # enforce_detection=False prevents it from crashing if the face is slightly turned
        objs = DeepFace.analyze(img_path = image_path, 
                                actions = ['emotion'],
                                enforce_detection=False)
        
        # DeepFace can find multiple faces, we just want the first one [0]
        result = objs[0]
        emotions = result['emotion']

        print("\nDetailed Score Breakdown:")
        # Sort emotions by score (highest first)
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        for emotion, score in sorted_emotions:
            print(f"  - {emotion.capitalize()}: {score:.2f}%")
            
    except Exception as e:
        print(f"Error analyzing face: {e}")

if __name__ == "__main__":
    # We need a test image!
    analyze_face("test_face.jpg")