#!/usr/bin/env python3
"""
BRSSD Dataset Download Script
Supports multiple sources: Roboflow, Kaggle, and GitHub
"""

import os
import sys
import argparse
from pathlib import Path

def download_from_roboflow(api_key=None, workspace="brssd", project="bangladeshi-road-signs", version=1):
    """Download BRSSD from Roboflow Universe"""
    try:
        from roboflow import Roboflow
        
        if not api_key:
            print("Please provide your Roboflow API key:")
            print("Visit: https://app.roboflow.com/settings/api")
            api_key = input("API Key: ").strip()
        
        rf = Roboflow(api_key=api_key)
        
        # List of potential BRSSD datasets on Roboflow
        candidates = [
            # (workspace, project, version)
            ("mostafinafis", "road-sign-detection-in-bd", 1),
            ("bangladesh-traffic-signs", "bangladesh-traffic-signs-v1", 1),
            ("thesis-kq02h", "bd-traffic-sign-detection", 1),
            (workspace, project, version) # Try the default last
        ]
        
        for ws, proj, ver in candidates:
            try:
                print(f"Attempting to download from Roboflow: {ws}/{proj} v{ver}...")
                project_obj = rf.workspace(ws).project(proj)
                dataset = project_obj.version(ver).download("yolov8", location="./BRSSD")
                print(f"✓ Successfully downloaded BRSSD from Roboflow ({ws}/{proj})")
                return True
            except Exception as e:
                print(f"  - Failed: {e}")
                continue
                
        print("✗ Could not find a valid BRSSD dataset on Roboflow with provided candidates.")
        return False

    except ImportError:
        print("⚠ Roboflow package not installed. Installing...")
        os.system("pip install roboflow -q")
        return download_from_roboflow(api_key, workspace, project, version)
    except Exception as e:
        print(f"✗ Error initializing Roboflow: {e}")
        return False

def download_from_kaggle(dataset_name="brssd"):
    """Download BRSSD from Kaggle"""
    try:
        import kaggle
        
        # Common BRSSD dataset names on Kaggle
        possible_datasets = [
            "fahadmehfoooz/brssd",
            "bangladesh-road-sign-dataset/brssd",
            "brssd/bangladeshi-road-signs"
        ]
        
        for dataset in possible_datasets:
            try:
                print(f"Attempting to download from Kaggle: {dataset}")
                os.system(f"kaggle datasets download -d {dataset} -p ./BRSSD --unzip")
                print(f"✓ Successfully downloaded BRSSD from Kaggle")
                return True
            except:
                continue
        
        print("✗ BRSSD not found on Kaggle with common names")
        print("Please search manually at: https://www.kaggle.com/datasets")
        return False
    except ImportError:
        print("⚠ Kaggle package not installed. Installing...")
        os.system("pip install kaggle -q")
        print("Please configure Kaggle API: https://www.kaggle.com/docs/api")
        return False
    except Exception as e:
        print(f"✗ Error downloading from Kaggle: {e}")
        return False

def download_from_github():
    """Download BRSSD from GitHub repositories"""
    repos = [
        "https://github.com/fahadmehfoooz/BRSSD",
        "https://github.com/BRSSD/dataset",
    ]
    
    for repo in repos:
        try:
            print(f"Attempting to clone from GitHub: {repo}")
            result = os.system(f"git clone {repo} ./BRSSD 2>/dev/null")
            if result == 0:
                print(f"✓ Successfully cloned BRSSD from GitHub")
                return True
        except:
            continue
    
    print("✗ Could not find BRSSD on GitHub")
    return False

def manual_download_instructions():
    """Provide manual download instructions"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nBRSSD Dataset Sources:")
    print("\n1. Roboflow Universe:")
    print("   - Visit: https://universe.roboflow.com/")
    print("   - Search for 'BRSSD' or 'Bangladesh Road Signs'")
    print("   - Download in YOLOv8 format")
    print("   - Extract to: ./BRSSD/")
    
    print("\n2. Kaggle:")
    print("   - Visit: https://www.kaggle.com/datasets")
    print("   - Search for 'BRSSD' or 'Bangladesh traffic signs'")
    print("   - Download and extract to: ./BRSSD/")
    
    print("\n3. Research Paper/Official Source:")
    print("   - Search for 'BRSSD dataset paper'")
    print("   - Contact authors for dataset access")
    
    print("\n4. GitHub:")
    print("   - Search: github.com/search?q=BRSSD")
    
    print("\nExpected directory structure:")
    print("  BRSSD/")
    print("  ├── train/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  ├── val/")
    print("  │   ├── images/")
    print("  │   └── labels/")
    print("  └── test/")
    print("      ├── images/")
    print("      └── labels/")
    print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Download BRSSD Dataset')
    parser.add_argument('--source', choices=['roboflow', 'kaggle', 'github', 'all'], 
                       default='all', help='Download source')
    parser.add_argument('--roboflow-api-key', help='Roboflow API key')
    args = parser.parse_args()
    
    print("BRSSD Dataset Downloader")
    print("="*60)
    
    success = False
    
    if args.source in ['roboflow', 'all']:
        print("\n[1/3] Trying Roboflow...")
        success = download_from_roboflow(args.roboflow_api_key)
        if success:
            return
    
    if args.source in ['kaggle', 'all'] and not success:
        print("\n[2/3] Trying Kaggle...")
        success = download_from_kaggle()
        if success:
            return
    
    if args.source in ['github', 'all'] and not success:
        print("\n[3/3] Trying GitHub...")
        success = download_from_github()
        if success:
            return
    
    if not success:
        print("\n✗ Automatic download failed from all sources")
        manual_download_instructions()

if __name__ == "__main__":
    main()
