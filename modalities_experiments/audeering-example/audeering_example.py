from test_audeering import Wav2Small
import librosa, torch
import sys
from vad_mapping import VADEmotionMapper, load_vad_prototypes

class EmotionPredictor:

    def __init__(self, device):
        self.device = device
        self.model = Wav2Small.from_pretrained(
            'audeering/wav2small').to(device).eval()
        self.vad_mapper = VADEmotionMapper(
            prototypes=load_vad_prototypes("vad_mapping.csv"),
            weights=(1.0, 1.0, 1.0), # Weights for VAD
            temperature=0.25, # Lower temp sharpens softmax probabilities
        )

    def predict(self, audio_path):
        signal = torch.from_numpy(librosa.load(audio_path, sr=16000)[0])[None, :]
        with torch.no_grad():
            logits = self.model(signal.to(self.device))
        arousal = logits[:, 0]
        dominance = logits[:, 1]
        valence = logits[:, 2]
        return arousal, dominance, valence
    
    def predict_ekman(self, audio_path):
        arousal, dominance, valence = self.predict(audio_path)
        vad_point = (valence.item(), arousal.item(), dominance.item())
        probabilities = self.vad_mapper.predict_proba(vad_point)
        return probabilities
    
if __name__ == "__main__":
    predictor = EmotionPredictor(device="cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) < 2:
        print("Usage: python audeering_example.py <audio_file.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    arousal, dominance, valence = predictor.predict(audio_path)
    print(f"Arousal: {arousal.item():.4f}, Dominance: {dominance.item():.4f}, Valence: {valence.item():.4f}")