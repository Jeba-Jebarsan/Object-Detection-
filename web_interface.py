import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
from collections import defaultdict, deque
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from PIL import Image
import json

# Set page config
st.set_page_config(
    page_title="🚀 Enhanced Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .detection-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

class WebDetectionSystem:
    def __init__(self):
        """Initialize web-based detection system"""
        if 'model' not in st.session_state:
            st.session_state.model = YOLO('yolov8n.pt')
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        if 'detection_stats' not in st.session_state:
            st.session_state.detection_stats = {
                'total_detections': 0,
                'humans': 0,
                'vehicles': 0,
                'animals': 0,
                'objects': 0
            }
        
        self.model = st.session_state.model
        self.confidence_threshold = 0.5
        
    def get_object_category(self, class_name):
        """Categorize detected objects"""
        vehicles = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane']
        animals = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        
        if class_name == 'person':
            return 'humans'
        elif class_name in vehicles:
            return 'vehicles'
        elif class_name in animals:
            return 'animals'
        else:
            return 'objects'
    
    def process_image(self, image):
        """Process uploaded image for detection"""
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = self.model(opencv_image, verbose=False)
        
        detections = []
        annotated_image = opencv_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    if confidence > self.confidence_threshold:
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Store detection
                        category = self.get_object_category(class_name)
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'category': category,
                            'bbox': bbox.tolist(),
                            'timestamp': time.time()
                        })
        
        # Convert back to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return annotated_image_rgb, detections
    
    def create_detection_chart(self, detections):
        """Create detection statistics chart"""
        if not detections:
            return None
        
        # Count detections by category
        category_counts = defaultdict(int)
        for detection in detections:
            category_counts[detection['category']] += 1
        
        # Create pie chart
        fig = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="Detections by Category",
            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c']
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return fig
    
    def create_confidence_chart(self, detections):
        """Create confidence distribution chart"""
        if not detections:
            return None
        
        confidences = [d['confidence'] for d in detections]
        
        fig = px.histogram(
            x=confidences,
            nbins=20,
            title="Detection Confidence Distribution",
            labels={'x': 'Confidence', 'y': 'Count'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
    
    def create_timeline_chart(self):
        """Create detection timeline chart"""
        if not st.session_state.detection_history:
            return None
        
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.detection_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Resample by minute and count
        timeline = df.set_index('timestamp').resample('1Min').size().reset_index()
        timeline.columns = ['Time', 'Detections']
        
        fig = px.line(
            timeline,
            x='Time',
            y='Detections',
            title="Detection Timeline",
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig

def main():
    """Main web application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🚀 Enhanced Object Detection System</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">Advanced AI-powered real-time object detection with modern web interface</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detection system
    detector = WebDetectionSystem()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Control Panel")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    detector.confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Detection mode
    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["📸 Image Upload", "📹 Webcam Stream", "📊 Analytics Dashboard"]
    )
    
    # Main content area
    if detection_mode == "📸 Image Upload":
        st.header("📸 Image Detection")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image for detection",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to detect objects, humans, and animals"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner("🔍 Processing image..."):
                annotated_image, detections = detector.process_image(image)
            
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_image, use_column_width=True)
            
            # Detection statistics
            if detections:
                st.success(f"✅ Found {len(detections)} objects!")
                
                # Update session state
                st.session_state.detection_history.extend(detections)
                for detection in detections:
                    category = detection['category']
                    st.session_state.detection_stats[category] += 1
                    st.session_state.detection_stats['total_detections'] += 1
                
                # Display detections
                st.subheader("🎯 Detected Objects")
                
                detection_cols = st.columns(min(len(detections), 4))
                for i, detection in enumerate(detections):
                    col_idx = i % 4
                    with detection_cols[col_idx]:
                        st.markdown(f"""
                        <div class="detection-card">
                            <h4>{detection['class'].title()}</h4>
                            <p><strong>Confidence:</strong> {detection['confidence']:.1%}</p>
                            <p><strong>Category:</strong> {detection['category']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    chart = detector.create_detection_chart(detections)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                
                with col2:
                    chart = detector.create_confidence_chart(detections)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
    
    elif detection_mode == "📹 Webcam Stream":
        st.header("📹 Live Webcam Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Stream Controls")
            
            if st.button("▶️ Start Stream", disabled=st.session_state.is_running):
                st.session_state.is_running = True
                st.success("Stream started!")
                st.rerun()
            
            if st.button("⏹️ Stop Stream", disabled=not st.session_state.is_running):
                st.session_state.is_running = False
                st.success("Stream stopped!")
                st.rerun()
            
            # Stream status
            if st.session_state.is_running:
                st.success("🟢 Stream Active")
            else:
                st.info("🔴 Stream Inactive")
            
            # Real-time stats
            st.subheader("📊 Live Statistics")
            stats = st.session_state.detection_stats
            
            st.markdown(f"""
            <div class="stat-card">
                <h4>Total Detections</h4>
                <h2>{stats['total_detections']}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            for category, count in stats.items():
                if category != 'total_detections':
                    st.metric(category.title(), count)
        
        with col1:
            if st.session_state.is_running:
                # Placeholder for webcam stream
                stream_placeholder = st.empty()
                
                # Note: In a real implementation, you would need to set up
                # a proper video streaming solution using WebRTC or similar
                stream_placeholder.info("🎥 Webcam stream would appear here\n\n(Requires additional WebRTC setup for live streaming)")
            else:
                st.info("Click 'Start Stream' to begin live detection")
    
    elif detection_mode == "📊 Analytics Dashboard":
        st.header("📊 Detection Analytics Dashboard")
        
        # Overview metrics
        st.subheader("📈 Overview")
        
        stats = st.session_state.detection_stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Detections", stats['total_detections'])
        with col2:
            st.metric("Humans", stats['humans'])
        with col3:
            st.metric("Vehicles", stats['vehicles'])
        with col4:
            st.metric("Animals", stats['animals'])
        with col5:
            st.metric("Objects", stats['objects'])
        
        # Charts
        if st.session_state.detection_history:
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                chart = detector.create_detection_chart(st.session_state.detection_history)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            with col2:
                # Timeline
                chart = detector.create_timeline_chart()
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Detection history table
            st.subheader("🗂️ Detection History")
            
            if st.session_state.detection_history:
                df = pd.DataFrame(st.session_state.detection_history)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df[['timestamp', 'class', 'category', 'confidence']].sort_values('timestamp', ascending=False)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export to CSV"):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"detection_history_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("🗑️ Clear History"):
                        st.session_state.detection_history = []
                        st.session_state.detection_stats = {
                            'total_detections': 0,
                            'humans': 0,
                            'vehicles': 0,
                            'animals': 0,
                            'objects': 0
                        }
                        st.success("History cleared!")
                        st.rerun()
        else:
            st.info("📭 No detection data available. Upload some images or run webcam detection to see analytics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin: 2rem 0;">
        🚀 Enhanced Object Detection System | Powered by YOLOv8 & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 