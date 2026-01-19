"""
CSV Export Module for Depression Detection Analysis
===================================================
Generates comprehensive CSV reports with all analysis data.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np


class AnalysisExporter:
    """
    Exports detailed analysis reports to CSV format.
    """
    
    def __init__(self, export_dir='analysis_reports'):
        """
        Initialize exporter.
        
        Args:
            export_dir: Directory to save CSV reports
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
        # Counter for auto-numbering
        self.report_counter = self._get_next_report_number()
        
        print(f"   AnalysisExporter initialized (output: {export_dir})")
    
    def _get_next_report_number(self):
        """Get the next report number for auto-numbering."""
        existing = list(self.export_dir.glob("Report_*.csv"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split('_')[1])
                numbers.append(num)
            except:
                pass
        return max(numbers) + 1 if numbers else 1
    
    def export_analysis(self, analysis_data, facial_features, voice_features):
        """
        Export complete analysis to CSV.
        
        Args:
            analysis_data: Dictionary with analysis results
            facial_features: Numpy array of shape (n_frames, 38)
            voice_features: Numpy array of shape (n_frames, 71)
        
        Returns:
            Path to saved CSV file
        """
        
        try:
            # Generate filename with auto-numbering
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Report_{self.report_counter}_{timestamp}.csv"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # ============================================================
                # SECTION 1: ANALYSIS METADATA
                # ============================================================
                writer.writerow(['ANALYSIS REPORT'])
                writer.writerow(['Report Number', self.report_counter])
                writer.writerow(['Timestamp', datetime.now().isoformat()])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 2: MODEL INFORMATION
                # ============================================================
                writer.writerow(['MODEL INFORMATION'])
                writer.writerow(['Facial Model', 'best_depression_model.keras'])
                writer.writerow(['Voice Model', 'voice_depression_detection_final.keras'])
                writer.writerow(['Facial Feature Scaler', 'feature_scaler.pkl'])
                writer.writerow(['Voice Feature Scaler', 'voice_feature_scaler.pkl'])
                writer.writerow(['Fusion Strategy', 'confidence_weighted'])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 3: OVERALL RESULTS
                # ============================================================
                writer.writerow(['OVERALL RESULTS'])
                writer.writerow(['Final Probability', f"{analysis_data['final_probability']:.4f}"])
                writer.writerow(['Final Probability %', f"{analysis_data['final_probability']*100:.2f}%"])
                writer.writerow(['Risk Level', analysis_data.get('risk_level', 'UNKNOWN')])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 4: MODALITY-SPECIFIC RESULTS
                # ============================================================
                writer.writerow(['FACIAL ANALYSIS RESULTS'])
                if analysis_data.get('facial_probability') is not None:
                    writer.writerow(['Facial Probability', f"{analysis_data['facial_probability']:.4f}"])
                    writer.writerow(['Facial Probability %', f"{analysis_data['facial_probability']*100:.2f}%"])
                    writer.writerow(['Facial Weight', f"{analysis_data['facial_weight']:.4f}"])
                else:
                    writer.writerow(['Facial Probability', 'N/A (Analysis Failed)'])
                writer.writerow([''])
                
                writer.writerow(['VOICE ANALYSIS RESULTS'])
                if analysis_data.get('voice_probability') is not None:
                    writer.writerow(['Voice Probability', f"{analysis_data['voice_probability']:.4f}"])
                    writer.writerow(['Voice Probability %', f"{analysis_data['voice_probability']*100:.2f}%"])
                    writer.writerow(['Voice Weight', f"{analysis_data['voice_weight']:.4f}"])
                else:
                    writer.writerow(['Voice Probability', 'N/A (Analysis Failed)'])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 5: FEATURE COUNTS
                # ============================================================
                writer.writerow(['FRAME AND FEATURE COUNTS'])
                writer.writerow(['Total Frames Extracted', analysis_data.get('total_frames', 'N/A')])
                writer.writerow(['Facial Features Shape', f"{facial_features.shape if facial_features is not None else 'N/A'}"])
                writer.writerow(['Voice Features Shape', f"{voice_features.shape if voice_features is not None else 'N/A'}"])
                writer.writerow(['Facial Feature Dimensions', 38])
                writer.writerow(['Voice Feature Dimensions', 71])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 6: FACIAL ACTION UNITS (Top Features)
                # ============================================================
                writer.writerow(['TOP FACIAL ACTION UNITS'])
                if analysis_data.get('feature_stats') and analysis_data['feature_stats'].get('top_action_units'):
                    writer.writerow(['Action Unit', 'Intensity', 'Intensity %'])
                    for au in analysis_data['feature_stats']['top_action_units']:
                        writer.writerow([
                            au['name'],
                            f"{au['intensity']:.4f}",
                            f"{au['intensity']*100:.2f}%"
                        ])
                else:
                    writer.writerow(['No AU data available'])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 7: VOICE FEATURE STATISTICS
                # ============================================================
                writer.writerow(['VOICE FEATURE STATISTICS'])
                if analysis_data.get('modality_insights') and analysis_data['modality_insights'].get('voice'):
                    voice_stats = analysis_data['modality_insights']['voice'].get('stats', {})
                    writer.writerow(['Metric', 'Value'])
                    for key, value in voice_stats.items():
                        if isinstance(value, (int, float)):
                            writer.writerow([key, f"{value:.4f}"])
                        else:
                            writer.writerow([key, str(value)])
                else:
                    writer.writerow(['No voice statistics available'])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 8: RAW FACIAL FEATURES (Frame-by-Frame)
                # ============================================================
                writer.writerow(['RAW FACIAL FEATURES (38 dimensions)'])
                writer.writerow(['Frame Number', 'AU1', 'AU2', 'AU3', 'AU4', 'AU5', 'AU6', 'AU7', 'AU8', 'AU9', 'AU10',
                                'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20',
                                'AU21', 'AU22', 'AU23', 'AU24', 
                                'Gaze1', 'Gaze2', 'Gaze3', 'Gaze4', 'Gaze5', 'Gaze6', 'Gaze7', 'Gaze8',
                                'Pose1', 'Pose2', 'Pose3', 'Pose4', 'Pose5', 'Pose6'])
                
                if facial_features is not None:
                    for frame_idx, frame_data in enumerate(facial_features):
                        row = [frame_idx + 1]
                        for value in frame_data:
                            row.append(f"{float(value):.6f}")
                        writer.writerow(row)
                writer.writerow([''])
                
                # ============================================================
                # SECTION 9: RAW VOICE FEATURES (Frame-by-Frame)
                # ============================================================
                writer.writerow(['RAW VOICE FEATURES (71 dimensions)'])
                writer.writerow(['Frame Number'] + [f'VoiceFeature_{i+1}' for i in range(71)])
                
                if voice_features is not None:
                    for frame_idx, frame_data in enumerate(voice_features):
                        row = [frame_idx + 1]
                        for value in frame_data:
                            row.append(f"{float(value):.6f}")
                        writer.writerow(row)
                writer.writerow([''])
                
                # ============================================================
                # SECTION 10: FEATURE DETAILS
                # ============================================================
                writer.writerow(['FEATURE DETAILS'])
                writer.writerow(['Feature Type', 'Description', 'Count'])
                writer.writerow(['Facial Action Units', 'Regression AUs (AU1-AU24)', 24])
                writer.writerow(['Gaze Features', 'Gaze direction (x,y,z) + confidence', 8])
                writer.writerow(['Pose Features', 'Head pose (Pitch, Roll, Yaw) + confidence', 6])
                writer.writerow(['MFCC Features', 'Mel-Frequency Cepstral Coefficients + deltas', 39])
                writer.writerow(['Chroma Features', 'Pitch-based features + deltas', 14])
                writer.writerow(['Spectral Features', 'Energy, centroid, rolloff, ZCR, flatness', 5])
                writer.writerow(['Pitch Features', 'Pitch contour + deltas from harmonic', 7])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 11: EXPLANATION
                # ============================================================
                writer.writerow(['CLINICAL EXPLANATION'])
                writer.writerow([''])
                explanation = analysis_data.get('explanation', 'No explanation available')
                for line in explanation.split('\n'):
                    writer.writerow([line])
                writer.writerow([''])
                
                # ============================================================
                # SECTION 12: MODALITY INSIGHTS
                # ============================================================
                writer.writerow(['DETAILED MODALITY INSIGHTS'])
                writer.writerow([''])
                
                if analysis_data.get('modality_insights'):
                    insights = analysis_data['modality_insights']
                    
                    if insights.get('facial'):
                        writer.writerow(['FACIAL MODALITY INSIGHTS'])
                        writer.writerow(['Probability', f"{insights['facial'].get('probability', 'N/A')}"])
                        writer.writerow(['Confidence', f"{insights['facial'].get('confidence', 'N/A')}"])
                        writer.writerow(['Assessment', insights['facial'].get('assessment', 'N/A')])
                        writer.writerow(['Signal Description', insights['facial'].get('signal', 'N/A')])
                        writer.writerow([''])
                    
                    if insights.get('voice'):
                        writer.writerow(['VOICE MODALITY INSIGHTS'])
                        writer.writerow(['Probability', f"{insights['voice'].get('probability', 'N/A')}"])
                        writer.writerow(['Confidence', f"{insights['voice'].get('confidence', 'N/A')}"])
                        writer.writerow(['Assessment', insights['voice'].get('assessment', 'N/A')])
                        writer.writerow(['Signal Description', insights['voice'].get('signal', 'N/A')])
                        writer.writerow([''])
                
                # ============================================================
                # SECTION 13: FOOTER
                # ============================================================
                writer.writerow(['END OF REPORT'])
                writer.writerow(['Generated', datetime.now().isoformat()])
                writer.writerow(['System', 'Multimodal Depression Detection System'])
            
            # Increment counter for next report
            self.report_counter += 1
            
            print(f"   ✅ Report saved: {filepath}")
            return str(filepath)
        
        except Exception as e:
            print(f"   ❌ Error exporting analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None