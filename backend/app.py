""" Flask Backend for Depression Detection Web Application ====================================================== This serves the web interface and handles video upload and real-time predictions with MULTIMODAL (Facial + Voice) support. """
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import json
import subprocess
import cv2
from pathlib import Path
import threading
import tempfile
import shutil

# Import our custom modules
from depression_model import DepressionPredictor
from openface_processor import OpenFaceProcessor
from audio_processor import AudioProcessor
from voice_predictor import VoicePredictor
from fusion_module import MultimodalFusion
from csv_export import AnalysisExporter

# Get the correct paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / 'frontend' / 'templates'
STATIC_DIR = BASE_DIR / 'frontend' / 'static'

# Initialize Flask app
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=120,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB buffer
    engineio_logger=False,
    logger=False
)

print(f"📁 Template folder: {TEMPLATE_DIR}")
print(f"📁 Static folder: {STATIC_DIR}")

# Paths
TEMP_DIR = Path('temp_openface')
UPLOADS_DIR = Path('uploads')
OPENFACE_EXE = r'C:\Users\Bimananto\Downloads\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe'
OPENFACE_MODEL = 'model/main_clnf_general.txt'

# Create directories
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL INITIALIZATION - FACIAL & VOICE
# ============================================================================
print("🔄 Loading models and processors...")

# Facial models
print("  → Loading facial model...")
predictor = DepressionPredictor(
    model_path='best_depression_model.keras',
    scaler_path='feature_scaler.pkl'
)
openface = OpenFaceProcessor(
    openface_path=OPENFACE_EXE,
    temp_dir=str(TEMP_DIR)
)

# Voice models
print("  → Loading voice model...")
voice_predictor = VoicePredictor(
    model_path='voice_depression_detection_final.keras',
    scaler_path='voice_feature_scaler.pkl'
)
audio_processor = AudioProcessor(temp_dir='temp_audio')

# Fusion module
print("  → Initializing fusion module...")
fusion = MultimodalFusion(fusion_strategy='confidence_weighted')

# CSV Export module
print("  → Initializing CSV export module...")
exporter = AnalysisExporter(export_dir='analysis_reports')

print("✅ All models loaded successfully!")

# Store session data
session_data = {}
prediction_history = {}

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'facial_model_loaded': predictor.model is not None,
        'voice_model_loaded': voice_predictor.model is not None,
    })

@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    client_id = request.sid
    session_data[client_id] = {
        'chunks': {},
        'total_chunks': 0
    }
    prediction_history[client_id] = []
    print(f"✅ Client connected: {client_id}")
    emit('response', {'data': 'Connected to server'}, broadcast=False)

@socketio.on('disconnect')
def handle_disconnect(auth=None):
    """Handle client disconnection."""
    client_id = request.sid
    cleanup_session(client_id)
    if client_id in session_data:
        del session_data[client_id]
    if client_id in prediction_history:
        del prediction_history[client_id]
    print(f"👋 Client disconnected: {client_id}")

@socketio.on_error_default
def default_error_handler(e):
    print(f"❌ Socket.IO error: {str(e)}")
    import traceback
    traceback.print_exc()

@socketio.on('upload_chunk')
def handle_video_chunk(data):
    """Receive video chunks from frontend and store them temporarily."""
    client_id = request.sid
    
    try:
        chunk_id = data.get('chunk')
        chunk_data = data.get('data')
        total_chunks = data.get('total_chunks')
        
        if client_id not in session_data:
            session_data[client_id] = {
                'chunks': {},
                'total_chunks': total_chunks
            }
        
        # Store chunk
        if chunk_data:
            if isinstance(chunk_data, bytes):
                session_data[client_id]['chunks'][chunk_id] = chunk_data
            else:
                session_data[client_id]['chunks'][chunk_id] = bytes(chunk_data)
        
        session_data[client_id]['total_chunks'] = total_chunks
        
        # Send acknowledgment
        socketio.emit('chunk_received', {
            'chunk': chunk_id,
            'total_chunks': total_chunks
        }, to=client_id)
        
        # Auto-start processing when all chunks received
        if len(session_data[client_id]['chunks']) == total_chunks:
            print(f"🎉 All {total_chunks} chunks received! Starting multimodal analysis...")
            socketio.emit('all_chunks_received', {'total_chunks': total_chunks}, to=client_id)
            
            # Start processing in background thread
            thread = threading.Thread(
                target=process_video,
                args=(client_id, total_chunks)
            )
            thread.daemon = True
            thread.start()
        
        return {'status': 'received', 'chunk': chunk_id}
        
    except Exception as e:
        print(f"❌ Error in upload_chunk handler: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f'Error receiving chunk: {str(e)}'}, to=client_id)
        return {'status': 'error', 'message': str(e)}

@socketio.on('process_video')
def handle_analyze(data):
    """Backup manual trigger."""
    client_id = request.sid
    try:
        total_chunks = data.get('total_chunks')
        
        if client_id not in session_data:
            return
        
        received = len(session_data[client_id]['chunks'])
        
        if received == total_chunks:
            thread = threading.Thread(
                target=process_video,
                args=(client_id, total_chunks)
            )
            thread.daemon = True
            thread.start()
        else:
            socketio.emit('error', {
                'message': f'Incomplete upload: {received}/{total_chunks} chunks received.',
                'error_type': 'incomplete_upload'
            }, to=client_id)
        
    except Exception as e:
        print(f"❌ Error in process_video: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f'Error starting analysis: {str(e)}'}, to=client_id)

def process_video(client_id, total_chunks):
    """
    Process video: reassemble chunks and analyze with both facial and voice models.
    Uses parallel processing for facial and voice analysis.
    """
    try:
        print(f"\n🎬 Starting MULTIMODAL analysis for client: {client_id}")
        
        if client_id not in session_data:
            raise Exception(f"Client {client_id} not in session_data")
        
        # Step 1: Reassemble video from chunks
        print("🔗 Reassembling video from chunks...")
        video_data = bytearray()
        
        for i in range(total_chunks):
            if i not in session_data[client_id]['chunks']:
                continue
            chunk = session_data[client_id]['chunks'][i]
            if isinstance(chunk, bytes):
                video_data.extend(chunk)
            elif isinstance(chunk, str):
                try:
                    video_data.extend(bytes.fromhex(chunk))
                except:
                    pass
        
        # Step 2: Save video file
        video_path = TEMP_DIR / f"video_{client_id}.webm"
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        file_size = len(video_data)
        print(f"💾 Video saved: {video_path} ({file_size} bytes)")
        
        if file_size == 0:
            raise Exception("Video file is empty - no data received from chunks")
        
        # Step 3: Extract frames from video
        socketio.emit('processing_update', {'status': 'Extracting frames from video...'}, to=client_id)
        frames = extract_frames(str(video_path))
        
        if not frames:
            raise Exception("No frames could be extracted from the video")
        
        print(f"📹 Extracted {len(frames)} frames from video")
        
        # ============================================================================
        # PARALLEL PROCESSING: FACIAL AND VOICE ANALYSIS
        # ============================================================================
        
        facial_result = {}
        voice_result = {}
        facial_error = None
        voice_error = None
        facial_features_array = None
        voice_features_array = None
        
        def analyze_facial():
            """Run facial analysis in thread."""
            nonlocal facial_result, facial_error, facial_features_array
            try:
                print("  → Running facial analysis...")
                socketio.emit('processing_update', 
                    {'status': f'Analyzing {len(frames)} frames with facial model...'}, 
                    to=client_id)
                
                features = extract_features_from_frames(frames)
                facial_features_array = features  # Store for CSV export
                if features is None or len(features) == 0:
                    raise Exception("Facial feature extraction failed")
                
                probability, explanation, feature_stats = predictor.predict(features)
                
                facial_result = {
                    'probability': float(probability),
                    'explanation': explanation,
                    'feature_stats': feature_stats,
                    'total_frames': len(frames),
                    'frames_analyzed': len(features)
                }
                print(f"  ✅ Facial analysis complete: {probability*100:.2f}%")
                
            except Exception as e:
                facial_error = str(e)
                print(f"  ❌ Facial analysis error: {str(e)}")
        
        def analyze_voice():
            """Run voice analysis in thread."""
            nonlocal voice_result, voice_error, voice_features_array
            try:
                print("  → Running voice analysis...")
                socketio.emit('processing_update', 
                    {'status': 'Extracting and analyzing audio features...'}, 
                    to=client_id)
                
                # Pass the original video path for better audio extraction
                voice_features = audio_processor.extract_features(frames, video_path=str(video_path))
                voice_features_array = voice_features  # Store for CSV export
                if voice_features is None:
                    raise Exception("Voice feature extraction failed")
                
                probability, confidence, explanation, stats = voice_predictor.predict(voice_features)
                
                voice_result = {
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'explanation': explanation,
                    'stats': stats
                }
                print(f"  ✅ Voice analysis complete: {probability*100:.2f}%")
                
            except Exception as e:
                voice_error = str(e)
                print(f"  ❌ Voice analysis error: {str(e)}")
        
        # Run both analyses in parallel
        print("⚙️  Running PARALLEL analysis...")
        facial_thread = threading.Thread(target=analyze_facial)
        voice_thread = threading.Thread(target=analyze_voice)
        
        facial_thread.start()
        voice_thread.start()
        
        facial_thread.join()
        voice_thread.join()
        
        print("✅ Both analyses completed!")
        
        # ============================================================================
        # FUSION AND RESULTS
        # ============================================================================
        
        socketio.emit('processing_update', {'status': 'Fusing results...'}, to=client_id)
        
        # Handle graceful degradation if one modality failed
        if facial_error and voice_error:
            raise Exception("Both facial and voice analysis failed")
        elif facial_error:
            print("⚠️  Using voice analysis only (facial failed)")
            final_probability = voice_result['probability']
            fusion_result = {
                'final_probability': final_probability,
                'facial_probability': None,
                'voice_probability': voice_result['probability'],
                'facial_weight': 0.0,
                'voice_weight': 1.0,
                'explanation': f"⚠️ Facial analysis failed. Using voice analysis only:\n\n{voice_result['explanation']}",
                'modality_insights': {'voice': voice_result},
                'risk_level': fusion.fusion_strategy
            }
        elif voice_error:
            print("⚠️  Using facial analysis only (voice failed)")
            final_probability = facial_result['probability']
            fusion_result = {
                'final_probability': final_probability,
                'facial_probability': facial_result['probability'],
                'voice_probability': None,
                'facial_weight': 1.0,
                'voice_weight': 0.0,
                'explanation': f"⚠️ Voice analysis failed. Using facial analysis only:\n\n{facial_result['explanation']}",
                'modality_insights': {'facial': facial_result},
                'risk_level': fusion.fusion_strategy
            }
        else:
            # Both succeeded - use fusion
            fusion_result = fusion.fuse(
                facial_result['probability'],
                0.85,  # Facial confidence (from training)
                voice_result['probability'],
                voice_result['confidence']
            )
            final_probability = fusion_result['final_probability']
        
        # Update history
        if client_id not in prediction_history:
            prediction_history[client_id] = []
        
        prediction_history[client_id].append({
            'probability': float(final_probability),
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 50 predictions
        if len(prediction_history[client_id]) > 50:
            prediction_history[client_id] = prediction_history[client_id][-50:]
        
        # Prepare final result
        result = {
            'status': 'success',
            'final_probability': float(final_probability),
            'facial_probability': float(facial_result.get('probability', 0)) if facial_result else None,
            'voice_probability': float(voice_result.get('probability', 0)) if voice_result else None,
            'facial_weight': fusion_result.get('facial_weight', 0),
            'voice_weight': fusion_result.get('voice_weight', 0),
            'risk_level': 'HIGH' if final_probability >= 0.7 else 'MODERATE' if final_probability >= 0.5 else 'LOW',
            'explanation': fusion_result.get('explanation', 'Analysis complete'),
            'feature_stats': facial_result.get('feature_stats', {}),
            'modality_insights': fusion_result.get('modality_insights', {}),
            'total_frames': len(frames),
            'history': prediction_history[client_id],
            'timestamp': datetime.now().isoformat()
        }
        
        # Export analysis to CSV (auto-save)
        print("📊 Exporting analysis to CSV...")
        csv_path = exporter.export_analysis(result, facial_features_array, voice_features_array)
        if csv_path:
            result['csv_report'] = csv_path
            print(f"✅ CSV report saved: {csv_path}")
        
        # Send result to client
        socketio.emit('analysis_complete', result, to=client_id)
        print(f"✅ Multimodal analysis complete: {final_probability*100:.2f}%")
        
        # Cleanup
        cleanup_session(client_id)
        
    except Exception as e:
        print(f"❌ Error in process_video: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {
            'message': f'Processing error: {str(e)}',
            'error_type': 'processing'
        }, to=client_id)
        cleanup_session(client_id)

def extract_frames(video_path, max_frames=1000):
    """Extract frames from video file with a reasonable limit."""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            # Limit to 300 frames (10 seconds at 30fps)
            if max_frames and frame_count >= max_frames:
                print(f"⚠️ Reached frame limit of {max_frames}")
                break
        
        cap.release()
        print(f"✅ Successfully extracted {len(frames)} frames")
        return frames
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        return []

def extract_features_from_frames(frames):
    """Extract facial features from frames using OpenFaceProcessor."""
    try:
        FRAME_SKIP = 1
        sampled_frames = frames[::FRAME_SKIP]
        
        print(f"📹 Processing {len(sampled_frames)} frames with OpenFace...")
        
        # Convert RGB frames to BGR for OpenFaceProcessor
        bgr_frames = []
        for frame in sampled_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr_frames.append(frame_bgr)
        
        # Use the OpenFaceProcessor to extract features
        features = openface.extract_features(bgr_frames)
        
        if features is None:
            print("❌ OpenFaceProcessor returned None")
            return None
        
        print(f"✅ OpenFace analysis completed: {features.shape}")
        return features
        
    except Exception as e:
        print(f"❌ Error extracting facial features: {e}")
        import traceback
        traceback.print_exc()
        return None

def cleanup_session(client_id):
    """Clean up temporary files for a session."""
    try:
        # Remove video files
        for pattern in ['video_*.webm', 'video_*.avi']:
            for f in TEMP_DIR.glob(pattern):
                try:
                    if client_id in str(f):
                        os.remove(f)
                except:
                    pass
        
        # Remove audio temp files
        try:
            import shutil
            audio_dir = Path('temp_audio')
            if audio_dir.exists():
                for f in audio_dir.glob('temp_audio_*'):
                    try:
                        os.remove(f)
                    except:
                        pass
        except:
            pass
        
        # Remove OpenFace output directories
        for d in TEMP_DIR.glob(f"openface_output_*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        
        # Clear session chunks
        if client_id in session_data:
            session_data[client_id]['chunks'].clear()
        
        print(f"🧹 Cleaned up session: {client_id}")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {str(e)}")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 MULTIMODAL DEPRESSION DETECTION WEB SERVER")
    print("="*70)
    print("\n📱 Access the application at:")
    print(" Local: http://localhost:5000")
    print(" Network: http://YOUR_IP:5000")
    print("\n🎤 Modalities: Facial + Voice")
    print("🔗 Fusion: Confidence-Weighted")
    print("\n⌨️ Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)