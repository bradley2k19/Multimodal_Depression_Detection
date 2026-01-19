"""
Voice Depression Prediction Model
==================================
Makes depression predictions from voice/audio features.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle


class VoicePredictor:
    """
    Handles depression prediction from voice features.
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize voice predictor with model and scaler.
        
        Args:
            model_path: Path to trained voice model (.keras)
            scaler_path: Path to voice feature scaler (.pkl)
        """
        print(f"   Loading voice model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        print(f"   Loading voice scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.sequence_length = 100
    
    def predict(self, features):
        """
        Make depression prediction from voice features.
        
        Args:
            features: numpy array of shape (100, 71) or (n_frames, 71)
        
        Returns:
            tuple: (probability, confidence, explanation, stats)
        """
        
        try:
            # Ensure we have exactly 100 frames
            if len(features) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
                features = np.vstack([features, padding])
            elif len(features) > self.sequence_length:
                features = features[-self.sequence_length:]
            
            # Normalize features using scaler
            features_scaled = self.scaler.transform(features)
            
            # Reshape for model input (1, 100, 71)
            features_input = features_scaled.reshape(1, self.sequence_length, features.shape[1])
            
            # Get prediction
            probability = float(self.model.predict(features_input, verbose=0)[0][0])
            
            # Ensure probability is in valid range
            probability = np.clip(probability, 0.0, 1.0)
            
            # Confidence is how certain the model is
            # (distance from 0.5, max = 0.5)
            confidence = min(abs(probability - 0.5) * 2, 0.95)
            
            # Generate explanation
            explanation = self._generate_explanation(probability)
            
            # Calculate feature statistics
            stats = self._calculate_feature_stats(features)
            
            return probability, confidence, explanation, stats
            
        except Exception as e:
            print(f"   ❌ Error in voice prediction: {str(e)}")
            return 0.5, 0.0, f"Error: {str(e)}", {}
    
    def _generate_explanation(self, probability):
        """Generate human-readable explanation from probability."""
        
        percentage = probability * 100
        
        if probability >= 0.7:
            explanation = f"🔴 High depression risk detected ({percentage:.1f}%). "
            explanation += "Voice analysis shows significant indicators of emotional distress. "
            explanation += "Speech patterns suggest reduced vocal variety and energy. "
            explanation += "Strong recommendation for professional mental health consultation."
        elif probability >= 0.5:
            explanation = f"🟡 Moderate depression risk detected ({percentage:.1f}%). "
            explanation += "Voice patterns show some indicators of emotional changes. "
            explanation += "Speech characteristics suggest possible emotional expression reduction. "
            explanation += "Consider professional consultation for comprehensive assessment."
        else:
            explanation = f"🟢 Low depression risk detected ({percentage:.1f}%). "
            explanation += "Voice analysis shows typical emotional expression patterns. "
            explanation += "Speech patterns appear normal with good vocal variety. "
            explanation += "Continue monitoring for any changes over time."
        
        return explanation
    
    def _calculate_feature_stats(self, features):
        """
        Calculate statistics about voice features.
        
        Returns:
            dict with feature insights
        """
        
        try:
            # Calculate means and stds of key features
            mfcc_features = features[:, :39]  # MFCC + deltas
            chroma_features = features[:, 39:53]  # Chroma
            spectral_features = features[:, 53:58]  # Spectral
            pitch_features = features[:, 58:]  # Pitch
            
            stats = {
                'mfcc_mean': float(np.mean(mfcc_features)),
                'mfcc_std': float(np.std(mfcc_features)),
                'chroma_mean': float(np.mean(chroma_features)),
                'spectral_centroid': float(np.mean(spectral_features[:, 0])) if spectral_features.shape[0] > 0 else 0.0,
                'pitch_std': float(np.std(pitch_features)),
                'pitch_mean': float(np.mean(pitch_features)),
                'total_frames': len(features),
                'feature_count': features.shape[1]
            }
            
            return stats
            
        except Exception as e:
            print(f"   ⚠️  Error calculating stats: {str(e)}")
            return {}