#!/usr/bin/env python3
"""
Advanced Model Manager for Object Detection System
Downloads and manages specialized models for different detection scenarios
"""

import os
import requests
import json
from pathlib import Path
import hashlib
from ultralytics import YOLO
import torch

class ModelManager:
    def __init__(self):
        """Initialize model manager with available models"""
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Available specialized models
        self.available_models = {
            'yolov10n': {
                'name': 'YOLOv10 Nano',
                'description': 'Latest YOLO with improved accuracy and speed',
                'size': '~6MB',
                'classes': 80,
                'dataset': 'COCO',
                'url': 'yolov10n.pt',
                'specialized_for': 'general_objects'
            },
            'yolov8n-world': {
                'name': 'YOLOv8 World Model',
                'description': 'Enhanced with WorldScope dataset (600+ classes)',
                'size': '~12MB',
                'classes': 600,
                'dataset': 'WorldScope',
                'url': 'yolov8n-world.pt',
                'specialized_for': 'diverse_objects'
            },
            'yolov8n-seg': {
                'name': 'YOLOv8 Instance Segmentation',
                'description': 'Provides pixel-level object segmentation',
                'size': '~7MB',
                'classes': 80,
                'dataset': 'COCO',
                'url': 'yolov8n-seg.pt',
                'specialized_for': 'precise_boundaries'
            },
            'animal_specialist': {
                'name': 'Animal Detection Specialist',
                'description': 'Fine-tuned for 50+ animal species',
                'size': '~15MB',
                'classes': 50,
                'dataset': 'AP-10K + Wildlife',
                'url': 'custom_animal_yolov8.pt',
                'specialized_for': 'animals'
            }
        }
        
        self.loaded_models = {}
    
    def list_available_models(self):
        """List all available models with descriptions"""
        print("🤖 Available Detection Models:")
        print("=" * 80)
        
        for model_key, model_info in self.available_models.items():
            status = "✅ Downloaded" if self.is_model_downloaded(model_key) else "⬇️  Available"
            print(f"\n{status} {model_info['name']}")
            print(f"   📋 {model_info['description']}")
            print(f"   📊 Classes: {model_info['classes']} | Size: {model_info['size']}")
            print(f"   🎯 Specialized for: {model_info['specialized_for']}")
            print(f"   📚 Dataset: {model_info['dataset']}")
    
    def is_model_downloaded(self, model_key):
        """Check if model is already downloaded"""
        if model_key not in self.available_models:
            return False
        
        model_path = self.models_dir / self.available_models[model_key]['url']
        return model_path.exists()
    
    def download_model(self, model_key, force_redownload=False):
        """Download a specific model"""
        if model_key not in self.available_models:
            print(f"❌ Model '{model_key}' not found in available models")
            return False
        
        model_info = self.available_models[model_key]
        model_path = self.models_dir / model_info['url']
        
        if model_path.exists() and not force_redownload:
            print(f"✅ Model '{model_info['name']}' already downloaded")
            return True
        
        print(f"⬇️  Downloading {model_info['name']}...")
        print(f"   📋 {model_info['description']}")
        print(f"   📦 Size: {model_info['size']}")
        
        try:
            # For standard YOLO models, let ultralytics handle the download
            if model_key.startswith('yolo'):
                model = YOLO(model_info['url'])
                print(f"✅ {model_info['name']} downloaded successfully!")
                return True
            else:
                # For custom models, would implement custom download logic
                print(f"⚠️  Custom model download not implemented yet for {model_key}")
                print("   Using fallback YOLOv8n model...")
                model = YOLO('yolov8n.pt')
                return True
                
        except Exception as e:
            print(f"❌ Failed to download {model_info['name']}: {e}")
            return False
    
    def load_model(self, model_key):
        """Load a model into memory"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if not self.is_model_downloaded(model_key):
            print(f"📥 Model not found locally, downloading {model_key}...")
            if not self.download_model(model_key):
                return None
        
        try:
            model_info = self.available_models[model_key]
            print(f"🔄 Loading {model_info['name']}...")
            
            model = YOLO(model_info['url'])
            self.loaded_models[model_key] = {
                'model': model,
                'info': model_info
            }
            
            print(f"✅ {model_info['name']} loaded successfully!")
            return self.loaded_models[model_key]
            
        except Exception as e:
            print(f"❌ Failed to load model {model_key}: {e}")
            return None
    
    def get_model_recommendations(self, use_case):
        """Get model recommendations based on use case"""
        recommendations = {
            'general': ['yolov10n', 'yolov8n-world'],
            'animals': ['animal_specialist', 'yolov8n-world'],
            'precision': ['yolov8n-seg', 'yolov10n'],
            'speed': ['yolov10n', 'yolov8n'],
            'diverse_objects': ['yolov8n-world', 'yolov10n']
        }
        
        return recommendations.get(use_case, ['yolov10n'])
    
    def optimize_for_scenario(self, scenario):
        """Recommend and load optimal models for specific scenarios"""
        scenarios = {
            'wildlife_monitoring': {
                'primary': 'animal_specialist',
                'secondary': 'yolov8n-world',
                'description': 'Optimized for detecting various animal species'
            },
            'security_surveillance': {
                'primary': 'yolov10n',
                'secondary': 'yolov8n-world',
                'description': 'Balanced speed and accuracy for real-time monitoring'
            },
            'research_analysis': {
                'primary': 'yolov8n-seg',
                'secondary': 'yolov8n-world',
                'description': 'High precision with detailed object boundaries'
            },
            'general_purpose': {
                'primary': 'yolov10n',
                'secondary': None,
                'description': 'Best overall performance for everyday use'
            }
        }
        
        if scenario not in scenarios:
            print(f"❌ Unknown scenario: {scenario}")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return None
        
        config = scenarios[scenario]
        print(f"🎯 Optimizing for: {scenario}")
        print(f"📋 {config['description']}")
        
        # Load primary model
        primary_model = self.load_model(config['primary'])
        if not primary_model:
            print(f"❌ Failed to load primary model for {scenario}")
            return None
        
        # Load secondary model if specified
        secondary_model = None
        if config['secondary']:
            secondary_model = self.load_model(config['secondary'])
        
        return {
            'scenario': scenario,
            'primary': primary_model,
            'secondary': secondary_model,
            'config': config
        }
    
    def get_system_info(self):
        """Get system information for model optimization"""
        system_info = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'cpu_count': os.cpu_count(),
            'models_downloaded': len([k for k in self.available_models.keys() if self.is_model_downloaded(k)])
        }
        
        print("💻 System Information:")
        print(f"   🖥️  GPU Available: {system_info['gpu_available']}")
        if system_info['gpu_available']:
            print(f"   🎮 GPU: {system_info['gpu_name']}")
        print(f"   🧮 CPU Cores: {system_info['cpu_count']}")
        print(f"   📦 Models Downloaded: {system_info['models_downloaded']}/{len(self.available_models)}")
        
        return system_info

def main():
    """Interactive model manager"""
    manager = ModelManager()
    
    print("🤖 Advanced Model Manager")
    print("=" * 50)
    
    while True:
        print("\n📋 Options:")
        print("1. List available models")
        print("2. Download a model")
        print("3. Optimize for scenario")
        print("4. System information")
        print("5. Exit")
        
        choice = input("\n🔤 Enter your choice (1-5): ").strip()
        
        if choice == '1':
            manager.list_available_models()
        
        elif choice == '2':
            manager.list_available_models()
            model_key = input("\n🔤 Enter model key to download: ").strip()
            manager.download_model(model_key)
        
        elif choice == '3':
            scenarios = ['wildlife_monitoring', 'security_surveillance', 'research_analysis', 'general_purpose']
            print("\n🎯 Available scenarios:")
            for i, scenario in enumerate(scenarios, 1):
                print(f"   {i}. {scenario}")
            
            try:
                scenario_idx = int(input("\n🔤 Choose scenario (1-4): ")) - 1
                if 0 <= scenario_idx < len(scenarios):
                    result = manager.optimize_for_scenario(scenarios[scenario_idx])
                    if result:
                        print(f"✅ System optimized for {result['scenario']}")
                else:
                    print("❌ Invalid scenario choice")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == '4':
            manager.get_system_info()
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 