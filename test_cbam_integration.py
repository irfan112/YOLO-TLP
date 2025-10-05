# from ultralytics import YOLO
# import torch

# print("Testing CBAM integration with YOLO...")

# # Test 1: Import check
# try:
#     from ultralytics.nn.modules.block import CBAM, CBAM_Lite
#     print("✅ CBAM modules imported successfully")
# except ImportError as e:
#     print(f"❌ Import failed: {e}")
#     exit(1)

# # Test 2: Load model with CBAM
# try:
#     model = YOLO('ultralytics/cfg/models/v12/yoloTLP.yaml')
#     print("✅ YOLO-TLP with CBAM loaded successfully")
#     print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")
# except Exception as e:
#     print(f"❌ Model loading failed: {e}")
#     exit(1)

# # Test 3: Forward pass
# try:
#     x = torch.randn(1, 3, 640, 640)
#     model.model.eval()
#     with torch.no_grad():
#         outputs = model.model(x)
    
#     # YOLO returns a list of outputs
#     if isinstance(outputs, (list, tuple)):
#         print("✅ Forward pass successful")
#         print(f"   Number of outputs: {len(outputs)}")
#         for i, out in enumerate(outputs):
#             if hasattr(out, 'shape'):
#                 print(f"   Output {i} shape: {out.shape}")
#     else:
#         print("✅ Forward pass successful")
#         print(f"   Output shape: {outputs.shape}")
        
# except Exception as e:
#     print(f"❌ Forward pass failed: {e}")
#     import traceback
#     traceback.print_exc()
#     exit(1)

# print("\n🎉 All tests passed! CBAM is ready to use.")
# print("\n📊 Model Summary:")
# print(f"   Parameters: 1,470,332 (~1.47M)")
# print(f"   Expected improvement over baseline:")
# print(f"   - bicycle: 0.062 → 0.10-0.12 mAP50")
# print(f"   - tricycle: 0.163 → 0.20-0.23 mAP50")
# print(f"   - Overall: 0.293 → 0.32-0.34 mAP50")


from ultralytics import YOLO
import torch

print("Testing SPDConv integration with YOLO...")

# Test 1: Import check
try:
    from ultralytics.nn.modules.block import SPDConv
    print("✅ SPDConv imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Fix: Add SPDConv class to block.py and 'SPDConv' to __all__")
    exit(1)

# Test 2: Module functionality
try:
    m = SPDConv(128, 256)
    x = torch.randn(1, 128, 144, 144)
    y = m(x)
    print(f"✅ SPDConv module works")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    assert y.shape == (1, 256, 72, 72), "Output shape incorrect"
except Exception as e:
    print(f"❌ Module test failed: {e}")
    exit(1)

# Test 3: Load YOLO-TLP v2 with SPDConv
try:
    model = YOLO('ultralytics/cfg/models/v12/yoloTLP.yaml')
    print("✅ YOLO-TLP v2 with SPDConv loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("Check if yolo-tlp-v2-novel.yaml exists and uses SPDConv correctly")
    exit(1)

# Test 4: Forward pass
try:
    x = torch.randn(1, 3, 640, 640)
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(x)
    print("✅ Forward pass successful")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    exit(1)

print("\n✅ All tests passed! SPDConv is ready.")