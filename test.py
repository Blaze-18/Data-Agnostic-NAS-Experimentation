#!/usr/bin/env python3
"""
GPU and Dependency Test Script
Tests for GPU availability and validates all key dependencies are working correctly.
"""

import sys
import platform

def test_python_version():
    """Test Python version compatibility."""
    print("=" * 60)
    print("🐍 PYTHON VERSION CHECK")
    print("=" * 60)
    version_info = sys.version_info
    print(f"Python Version: {sys.version}")
    print(f"Version Info: {version_info.major}.{version_info.minor}.{version_info.micro}")
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        print("⚠️  WARNING: Python 3.8+ is recommended")
        return False
    print("✅ Python version check passed\n")
    return True

def test_pytorch():
    """Test PyTorch installation and GPU availability."""
    print("=" * 60)
    print("🔥 PYTORCH CHECK")
    print("=" * 60)
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"PyTorch location: {torch.__file__}")
        
        # Check CUDA availability
        print("\n📊 CUDA & GPU Status:")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"  - cuDNN available: {torch.backends.cudnn.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  - GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        else:
            print("  ⚠️  No GPU detected. Running on CPU (slower).")
        
        # Test simple tensor operation
        print("\n⚙️  Testing tensor operations:")
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print(f"  - CPU tensor operation successful (result shape: {z.shape})")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print(f"  - GPU tensor operation successful (result shape: {z_gpu.shape})")
        
        print("✅ PyTorch check passed\n")
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch test error: {e}")
        return False

def test_numpy():
    """Test NumPy installation."""
    print("=" * 60)
    print("📐 NUMPY CHECK")
    print("=" * 60)
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test basic operations
        arr = np.random.rand(1000, 1000)
        result = np.dot(arr, arr.T)
        print(f"  - Matrix multiplication successful (result shape: {result.shape})")
        
        print("✅ NumPy check passed\n")
        return True
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ NumPy test error: {e}")
        return False

def test_scipy():
    """Test SciPy installation."""
    print("=" * 60)
    print("🔬 SCIPY CHECK")
    print("=" * 60)
    try:
        import scipy
        from scipy import stats
        print(f"✅ SciPy version: {scipy.__version__}")
        
        # Test Spearman correlation
        import numpy as np
        x = np.random.rand(100)
        y = np.random.rand(100)
        rho, p = stats.spearmanr(x, y)
        print(f"  - Spearman correlation successful (ρ={rho:.4f}, p={p:.4f})")
        
        print("✅ SciPy check passed\n")
        return True
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ SciPy test error: {e}")
        return False

def test_scikit_learn():
    """Test scikit-learn installation."""
    print("=" * 60)
    print("🤖 SCIKIT-LEARN CHECK")
    print("=" * 60)
    try:
        import sklearn
        print(f"✅ scikit-learn version: {sklearn.__version__}")
        
        # Test basic functionality
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import numpy as np
        
        X = np.random.rand(100, 20)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"  - StandardScaler successful (output shape: {X_scaled.shape})")
        
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)
        print(f"  - PCA successful (output shape: {X_pca.shape})")
        
        print("✅ scikit-learn check passed\n")
        return True
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ scikit-learn test error: {e}")
        return False

def test_h5py():
    """Test h5py installation."""
    print("=" * 60)
    print("💾 H5PY CHECK")
    print("=" * 60)
    try:
        import h5py
        print(f"✅ h5py version: {h5py.__version__}")
        print(f"  - HDF5 lib version: {h5py.version.hdf5_version}")
        
        print("✅ h5py check passed\n")
        return True
    except ImportError as e:
        print(f"❌ h5py import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ h5py test error: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow installation (needed for NAS-Bench-101)."""
    print("=" * 60)
    print("🧠 TENSORFLOW CHECK")
    print("=" * 60)
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check if tf.compat.v1 is available (needed for nasbench)
        print(f"  - tf.compat.v1 available: {hasattr(tf, 'compat')}")
        if hasattr(tf, 'compat'):
            print(f"  - tf.compat.v1.io available: {hasattr(tf.compat.v1, 'io')}")
        
        print("✅ TensorFlow check passed\n")
        return True
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        print("  ℹ️  Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ TensorFlow test error: {e}")
        return False

def test_protobuf():
    """Test Protocol Buffers installation."""
    print("=" * 60)
    print("📦 PROTOBUF CHECK")
    print("=" * 60)
    try:
        import google.protobuf
        from google.protobuf import __version__ as pb_version
        print(f"✅ Protobuf version: {pb_version}")
        
        # Check environment variable
        import os
        impl = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "not set")
        print(f"  - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: {impl}")
        if impl != "python":
            print("  ⚠️  WARNING: Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python for nasbench")
        
        print("✅ Protobuf check passed\n")
        return True
    except ImportError as e:
        print(f"❌ Protobuf import failed: {e}")
        print("  ℹ️  Install with: pip install protobuf==3.20.3")
        return False
    except Exception as e:
        print(f"❌ Protobuf test error: {e}")
        return False

def test_nasbench():
    """Test NAS-Bench package installation."""
    print("=" * 60)
    print("🏗️  NASBENCH CHECK")
    print("=" * 60)
    try:
        import os
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        
        from nasbench import api
        print(f"✅ NAS-Bench package imported successfully")
        print(f"  - Module location: {api.__file__}")
        
        # Check if TFRecord exists
        from pathlib import Path
        tfrecord_path = Path("data/nasbench101/nasbench_full.tfrecord")
        if tfrecord_path.exists():
            size_gb = tfrecord_path.stat().st_size / 1e9
            print(f"  - TFRecord found: {tfrecord_path} ({size_gb:.2f} GB)")
        else:
            print(f"  ⚠️  TFRecord not found at: {tfrecord_path}")
            print("     (This is OK if you haven't downloaded it yet)")
        
        print("✅ NAS-Bench check passed\n")
        return True
    except ImportError as e:
        print(f"❌ NAS-Bench import failed: {e}")
        print("  ℹ️  Install with: pip install nasbench")
        print("  ℹ️  You may also need to apply API patches (see LAB_MACHINE_CONTEXT_NASBENCH101.md)")
        return False
    except Exception as e:
        print(f"❌ NAS-Bench test error: {e}")
        return False

def test_matplotlib():
    """Test Matplotlib installation."""
    print("=" * 60)
    print("📊 MATPLOTLIB CHECK")
    print("=" * 60)
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib version: {matplotlib.__version__}")
        print(f"  - Backend: {matplotlib.get_backend()}")
        
        print("✅ Matplotlib check passed\n")
        return True
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Matplotlib test error: {e}")
        return False

def test_system_info():
    """Display system information."""
    print("=" * 60)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print()

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  GPU & DEPENDENCY VALIDATION TEST".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    test_system_info()
    
    results = {
        "Python Version": test_python_version(),
        "PyTorch": test_pytorch(),
        "NumPy": test_numpy(),
        "SciPy": test_scipy(),
        "scikit-learn": test_scikit_learn(),
        "Matplotlib": test_matplotlib(),
        "TensorFlow": test_tensorflow(),
        "Protobuf": test_protobuf(),
        "NAS-Bench": test_nasbench(),
    }
    
    # Summary
    print("=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(results.values())
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready to go!\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
