"""
Audio Feature Extraction for Voice Depression Detection
=======================================================
Extracts acoustic features from video audio streams.
"""

import cv2
import numpy as np
import librosa
import os
from pathlib import Path

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("   ⚠️  PyAV not installed, audio extraction may fail. Install with: pip install av")


class AudioProcessor:
    """
    Processes audio from video files and extracts acoustic features.
    """
    
    def __init__(self, temp_dir='temp_audio'):
        """
        Initialize audio processor.
        
        Args:
            temp_dir: Directory for temporary audio files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.audio_counter = 0
        
        # Audio parameters (matching training)
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.sequence_length = 1000
        
        print(f"   AudioProcessor initialized (temp_dir: {temp_dir})")
    
    def extract_features(self, frames, video_path=None):
        """
        Extract audio features from video.
        
        Args:
            frames: List of numpy arrays (video frames)
            video_path: Path to original video file (optional, for better audio extraction)
        
        Returns:
            numpy array of shape (100, 71) or None if failed
        """
        try:
            print("   Extracting audio from video...")
            
            # If we have the original video path, use that (it has audio)
            # Otherwise, create a temporary video from frames
            if video_path and os.path.exists(video_path):
                print(f"   Using original video file for audio extraction: {video_path}")
                audio = self._extract_audio_from_video_file(video_path)
            else:
                print("   No video file path provided, using frames...")
                audio = self._extract_audio_from_frames(frames)
            
            if audio is None or len(audio) == 0:
                print(f"   ❌ Failed to extract audio")
                return None
            
            print(f"   ✅ Audio extracted: {len(audio)} samples")
            
            # Extract features from audio
            print("   Extracting features from audio...")
            features = self._extract_audio_features(audio)
            
            # Normalize features to 100 frames
            if features is not None:
                features = self._normalize_sequence_length(features)
            
            return features
            
        except Exception as e:
            print(f"   ❌ Error in extract_features: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_audio_from_video_file(self, video_path):
        """
        Extract audio directly from a video file using PyAV.
        This is much more reliable than creating a temporary video.
        """
        try:
            if HAS_PYAV:
                print(f"   Extracting audio using PyAV from: {video_path}")
                container = av.open(video_path)
                
                # Find audio stream
                audio_stream = None
                for stream in container.streams:
                    if stream.type == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream is None:
                    print(f"   ⚠️  No audio stream in video file")
                    return None
                
                # Extract audio samples
                audio_data = []
                for frame in container.decode(audio_stream):
                    audio_data.append(frame.to_ndarray())
                
                if not audio_data:
                    print(f"   ⚠️  No audio frames extracted")
                    return None
                
                # Concatenate all audio frames
                y = np.concatenate(audio_data, axis=-1).flatten()
                
                # Resample to 16kHz if needed
                if audio_stream.sample_rate != self.sample_rate:
                    print(f"   Resampling from {audio_stream.sample_rate}Hz to {self.sample_rate}Hz")
                    import scipy.signal
                    num_samples = int(len(y) * self.sample_rate / audio_stream.sample_rate)
                    y = scipy.signal.resample(y, num_samples)
                
                # Convert to mono if stereo
                if len(y.shape) > 1:
                    y = np.mean(y, axis=0)
                
                # Normalize
                y_max = np.max(np.abs(y))
                if y_max > 0:
                    y = y / y_max
                
                print(f"   ✅ Audio extracted via PyAV: {len(y)} samples")
                return y
            
            else:
                print(f"   ⚠️  PyAV not available, trying librosa...")
                y, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
                print(f"   ✅ Audio loaded via librosa: {len(y)} samples")
                return y
                
        except Exception as e:
            print(f"   ⚠️  Error extracting audio from file: {str(e)}")
            return None
    
    def _extract_audio_from_frames(self, frames):
        """
        Extract audio directly from video frames without ffmpeg.
        Uses PyAV for robust video/audio handling.
        """
        try:
            self.audio_counter += 1
            temp_video = self.temp_dir / f"temp_audio_video_{self.audio_counter}.avi"
            
            # Save frames as temporary video
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(str(temp_video), fourcc, 30, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            print(f"   📹 Saved {len(frames)} frames to temporary video: {temp_video}")
            
            # Try to extract audio using PyAV first
            if HAS_PYAV:
                try:
                    print(f"   Extracting audio using PyAV...")
                    container = av.open(str(temp_video))
                    
                    # Get audio stream
                    audio_stream = None
                    for stream in container.streams:
                        if stream.type == 'audio':
                            audio_stream = stream
                            break
                    
                    if audio_stream is None:
                        print(f"   ⚠️  No audio stream found in video")
                        return None
                    
                    # Extract audio samples
                    audio_data = []
                    for frame in container.decode(audio_stream):
                        audio_data.append(frame.to_ndarray())
                    
                    if not audio_data:
                        print(f"   ⚠️  No audio frames extracted")
                        return None
                    
                    # Concatenate all audio frames
                    y = np.concatenate(audio_data, axis=1).flatten()
                    
                    # Resample to 16kHz if needed
                    if audio_stream.sample_rate != self.sample_rate:
                        print(f"   Resampling from {audio_stream.sample_rate}Hz to {self.sample_rate}Hz")
                        import scipy.signal
                        num_samples = int(len(y) * self.sample_rate / audio_stream.sample_rate)
                        y = scipy.signal.resample(y, num_samples)
                    
                    # Convert to mono if stereo
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=0)
                    
                    # Normalize
                    y_max = np.max(np.abs(y))
                    if y_max > 0:
                        y = y / y_max
                    
                    print(f"   ✅ Audio extracted: {len(y)} samples")
                    
                    # Cleanup
                    try:
                        os.remove(str(temp_video))
                    except:
                        pass
                    
                    return y
                    
                except Exception as e:
                    print(f"   ⚠️  PyAV extraction failed: {str(e)}")
            
            # Fallback: try librosa
            print(f"   Trying librosa fallback...")
            try:
                y, sr = librosa.load(str(temp_video), sr=self.sample_rate, mono=True)
                print(f"   ✅ Audio loaded: {len(y)} samples at {sr}Hz")
                
                # Cleanup
                try:
                    os.remove(str(temp_video))
                except:
                    pass
                
                return y
                
            except Exception as e:
                print(f"   ⚠️  Librosa also failed: {str(e)}")
                
                # Cleanup
                try:
                    os.remove(str(temp_video))
                except:
                    pass
                
                return None
            
        except Exception as e:
            print(f"   ❌ Error extracting audio from frames: {str(e)}")
            return None
    
    def _extract_audio_features(self, audio):
        """
        Extract acoustic features from audio signal.
        
        Args:
            audio: numpy array of audio samples
        
        Returns:
            numpy array of shape (n_frames, 71) with features
        """
        
        try:
            if len(audio) < self.sample_rate:
                print(f"   ⚠️  Audio too short ({len(audio)} samples)")
                return None
            
            # Normalize audio
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / audio_max
            
            print(f"   Computing MFCC features...")
            features_list = []
            
            # 1. MFCCs (39: 13 + 13 delta + 13 delta-delta)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features_list.append(mfcc)
            features_list.append(mfcc_delta)
            features_list.append(mfcc_delta2)
            print(f"   ✓ MFCC computed")
            
            # 2. Chroma features (14: 12 chroma + 2 extra)
            print(f"   Computing Chroma features...")
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_delta = librosa.feature.delta(chroma)
            features_list.append(chroma)
            features_list.append(chroma_delta)
            print(f"   ✓ Chroma computed")
            
            # 3. Spectral features (5)
            print(f"   Computing Spectral features...")
            S = np.abs(librosa.stft(audio))
            spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sample_rate)
            zero_crossing = librosa.feature.zero_crossing_rate(audio)
            rms = librosa.feature.rms(y=audio)
            spectral_flatness = librosa.feature.spectral_flatness(S=S)
            
            features_list.append(spectral_centroid)
            features_list.append(spectral_rolloff)
            features_list.append(zero_crossing)
            features_list.append(rms)
            features_list.append(spectral_flatness)
            print(f"   ✓ Spectral features computed")
            
            # 4. Pitch features (7: from harmonic component)
            print(f"   Computing Pitch features...")
            harmonic, _ = librosa.effects.hpss(audio)
            S_h = np.abs(librosa.stft(harmonic))
            pitch_centroid = librosa.feature.spectral_centroid(S=S_h, sr=self.sample_rate)
            pitch_delta = librosa.feature.delta(pitch_centroid)
            pitch_delta2 = librosa.feature.delta(pitch_centroid, order=2)
            features_list.append(pitch_centroid)
            features_list.append(pitch_delta)
            features_list.append(pitch_delta2)
            print(f"   ✓ Pitch features computed")
            
            # Combine all features (frames × features)
            print(f"   Combining all features...")
            combined = np.concatenate(features_list, axis=0)  # (71, n_frames)
            combined = combined.T  # (n_frames, 71)
            
            print(f"   🎵 Extracted features: shape {combined.shape}")
            
            return combined
            
        except Exception as e:
            print(f"   ❌ Error extracting features: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _normalize_sequence_length(self, features):
        """
        Pad or trim features to exactly 100 frames.
        
        Args:
            features: numpy array of shape (n_frames, 71)
        
        Returns:
            numpy array of shape (100, 71)
        """
        
        if len(features) == self.sequence_length:
            return features
        elif len(features) < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])
            print(f"   📊 Padded features to {len(features)} frames")
        else:
            # Take last 100 frames
            features = features[-self.sequence_length:]
            print(f"   📊 Trimmed features to last {len(features)} frames")
        
        return features.astype(np.float32)
    