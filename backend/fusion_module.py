"""
Multimodal Fusion Module
========================
Intelligently combines facial and voice predictions.
"""

import numpy as np


class MultimodalFusion:
    """
    Fuses facial and voice depression predictions using confidence weighting.
    """
    
    def __init__(self, fusion_strategy='confidence_weighted'):
        """
        Initialize fusion module.
        
        Args:
            fusion_strategy: Strategy for combining predictions
                - 'confidence_weighted': Weight by model confidence (default)
                - 'simple_average': Equal 50/50 weight
                - 'voice_emphasis': 60% voice, 40% facial
                - 'max_probability': Take the higher score (conservative)
        """
        self.fusion_strategy = fusion_strategy
        print(f"   Fusion strategy: {fusion_strategy}")
    
    def fuse(self, facial_prob, facial_conf, voice_prob, voice_conf):
        """
        Combine facial and voice predictions.
        
        Args:
            facial_prob: Facial depression probability (0-1)
            facial_conf: Facial model confidence (0-1)
            voice_prob: Voice depression probability (0-1)
            voice_conf: Voice model confidence (0-1)
        
        Returns:
            dict with:
            - final_probability: Fused prediction (0-1)
            - facial_weight: Weight given to facial
            - voice_weight: Weight given to voice
            - explanation: Detailed analysis
            - modality_insights: Per-modality information
        """
        
        try:
            if self.fusion_strategy == 'confidence_weighted':
                final_prob, facial_w, voice_w = self._fuse_confidence_weighted(
                    facial_prob, facial_conf, voice_prob, voice_conf
                )
            elif self.fusion_strategy == 'simple_average':
                final_prob = (facial_prob + voice_prob) / 2
                facial_w = 0.5
                voice_w = 0.5
            elif self.fusion_strategy == 'voice_emphasis':
                final_prob = facial_prob * 0.4 + voice_prob * 0.6
                facial_w = 0.4
                voice_w = 0.6
            elif self.fusion_strategy == 'max_probability':
                final_prob = max(facial_prob, voice_prob)
                facial_w = 1.0 if facial_prob >= voice_prob else 0.0
                voice_w = 1.0 if voice_prob > facial_prob else 0.0
            else:
                # Default to confidence weighted
                final_prob, facial_w, voice_w = self._fuse_confidence_weighted(
                    facial_prob, facial_conf, voice_prob, voice_conf
                )
            
            # Ensure valid range
            final_prob = np.clip(final_prob, 0.0, 1.0)
            
            # Generate explanation
            explanation = self._generate_explanation(
                final_prob, facial_prob, voice_prob, facial_w, voice_w
            )
            
            # Get modality insights
            modality_insights = self._get_modality_insights(
                facial_prob, voice_prob, facial_conf, voice_conf
            )
            
            result = {
                'final_probability': final_prob,
                'facial_probability': facial_prob,
                'voice_probability': voice_prob,
                'facial_weight': facial_w,
                'voice_weight': voice_w,
                'facial_confidence': facial_conf,
                'voice_confidence': voice_conf,
                'explanation': explanation,
                'modality_insights': modality_insights,
                'risk_level': self._get_risk_level(final_prob)
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error in fusion: {str(e)}")
            return {
                'final_probability': 0.5,
                'facial_probability': facial_prob,
                'voice_probability': voice_prob,
                'facial_weight': 0.5,
                'voice_weight': 0.5,
                'explanation': f"Fusion error: {str(e)}",
                'modality_insights': {},
                'risk_level': 'UNKNOWN'
            }
    
    def _fuse_confidence_weighted(self, facial_prob, facial_conf, voice_prob, voice_conf):
        """
        Smart fusion: weight each modality by its confidence.
        
        Models with higher confidence contribute more to final decision.
        """
        
        total_confidence = facial_conf + voice_conf
        
        # Avoid division by zero
        if total_confidence < 0.01:
            facial_weight = 0.5
            voice_weight = 0.5
        else:
            facial_weight = facial_conf / total_confidence
            voice_weight = voice_conf / total_confidence
        
        final_probability = (facial_prob * facial_weight) + (voice_prob * voice_weight)
        
        return final_probability, facial_weight, voice_weight
    
    def _generate_explanation(self, final_prob, facial_prob, voice_prob, facial_w, voice_w):
        """Generate detailed multimodal explanation."""
        
        percentage = final_prob * 100
        
        # Overall risk assessment
        if final_prob >= 0.7:
            overall = f"🔴 HIGH RISK ({percentage:.1f}%)"
            risk_text = "strong indicators of depression"
        elif final_prob >= 0.5:
            overall = f"🟡 MODERATE RISK ({percentage:.1f}%)"
            risk_text = "moderate indicators of depression"
        else:
            overall = f"🟢 LOW RISK ({percentage:.1f}%)"
            risk_text = "minimal depression indicators"
        
        # Agreement analysis
        facial_pct = facial_prob * 100
        voice_pct = voice_prob * 100
        agreement = abs(facial_prob - voice_prob)
        
        if agreement < 0.15:
            agreement_text = "✓ Facial and voice models STRONGLY AGREE"
        elif agreement < 0.3:
            agreement_text = "~ Facial and voice models AGREE"
        else:
            agreement_text = "⚠ Facial and voice models DISAGREE"
        
        # Build explanation
        explanation = f"{overall}\n\n"
        explanation += f"📊 Analysis Breakdown:\n"
        explanation += f"  👁️ Facial: {facial_pct:.1f}% (weight: {facial_w*100:.1f}%)\n"
        explanation += f"  🎤 Voice: {voice_pct:.1f}% (weight: {voice_w*100:.1f}%)\n\n"
        explanation += f"{agreement_text}\n\n"
        explanation += f"The multimodal analysis shows {risk_text}.\n"
        
        if final_prob >= 0.6:
            explanation += "🔴 Professional mental health consultation is strongly recommended."
        elif final_prob >= 0.4:
            explanation += "🟡 Consider professional consultation for comprehensive assessment."
        else:
            explanation += "🟢 Continue monitoring and maintain healthy lifestyle practices."
        
        return explanation
    
    def _get_modality_insights(self, facial_prob, voice_prob, facial_conf, voice_conf):
        """Get insights about what each modality detected."""
        
        facial_pct = facial_prob * 100
        voice_pct = voice_prob * 100
        
        insights = {
            'facial': {
                'probability': facial_prob,
                'confidence': facial_conf,
                'percentage': facial_pct,
                'assessment': self._probability_to_assessment(facial_prob),
                'signal': 'Facial expressions and micro-movements indicate emotional state'
            },
            'voice': {
                'probability': voice_prob,
                'confidence': voice_conf,
                'percentage': voice_pct,
                'assessment': self._probability_to_assessment(voice_prob),
                'signal': 'Voice characteristics including pitch, energy, and prosody indicate emotional state'
            }
        }
        
        return insights
    
    def _probability_to_assessment(self, probability):
        """Convert probability to text assessment."""
        
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.5:
            return "Moderate Risk"
        elif probability >= 0.3:
            return "Low Risk"
        else:
            return "Minimal Risk"
    
    def _get_risk_level(self, probability):
        """Get categorical risk level."""
        
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.5:
            return "MODERATE"
        elif probability >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"