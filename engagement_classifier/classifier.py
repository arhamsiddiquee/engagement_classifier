import joblib
import pandas as pd
import os

class EngagementClassifier:
    def __init__(self, model_path='models/engagement_svm_model.pkl', scaler_path='models/scaler.pkl'):
        model_dir = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(model_dir, model_path))
        self.scaler = joblib.load(os.path.join(model_dir, scaler_path))

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# Example usage within the module
if __name__ == "__main__":
    clf = EngagementClassifier()
    # Example input
    X_new = pd.DataFrame({
        ' gaze_angle_x': [0.0],
        ' gaze_angle_y': [0.0],
        ' pose_Tx': [0.0],
        ' pose_Ty': [0.0],
        ' pose_Tz': [0.0],
        ' pose_Rx': [0.0],
        ' pose_Ry': [0.0],
        ' pose_Rz': [0.0],
        ' AU01_r': [0.0],
        ' AU02_r': [0.0],
        ' AU04_r': [0.0],
        ' AU05_r': [0.0],
        ' AU06_r': [0.0],
        ' AU07_r': [0.0],
        ' AU09_r': [0.0],
        ' AU10_r': [0.0],
        ' AU12_r': [0.0],
        ' AU14_r': [0.0],
        ' AU15_r': [0.0],
        ' AU17_r': [0.0],
        ' AU20_r': [0.0],
        ' AU23_r': [0.0],
        ' AU25_r': [0.0],
        ' AU26_r': [0.0],
        ' AU45_r': [0.0],
        ' AU01_c': [0.0],
        ' AU02_c': [0.0],
        ' AU04_c': [0.0],
        ' AU05_c': [0.0],
        ' AU06_c': [0.0],
        ' AU07_c': [0.0],
        ' AU09_c': [0.0],
        ' AU10_c': [0.0],
        ' AU12_c': [0.0],
        ' AU14_c': [0.0],
        ' AU15_c': [0.0],
        ' AU17_c': [0.0],
        ' AU20_c': [0.0],
        ' AU23_c': [0.0],
        ' AU25_c': [0.0],
        ' AU26_c': [0.0],
        ' AU28_c': [0.0],
        ' AU45_c': [0.0]
    })

    print(clf.predict(X_new))
