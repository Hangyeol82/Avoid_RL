
import numpy as np
import torch
from env.dyn_env_one import DynAvoidOneObjEnv
from rl.network import ActorCritic

def test_cnn_integration():
    print("=== Testing CNN Integration ===")
    
    # 1. Create Environment
    print("[1] Creating Environment...")
    grid = np.zeros((30, 30), dtype=np.int8)
    # Add some walls
    grid[10:20, 10:20] = 1
    
    wps = np.array([[5, 5], [25, 25]], dtype=np.float32)
    
    env = DynAvoidOneObjEnv(
        grid=grid,
        waypoints=wps,
        local_map_size=15  # Check if this arg works
    )
    
    obs, _ = env.reset()
    print(f"Environment Created. Obs shape: {obs.shape}")
    
    # Check obs dimension
    # Expected: MLP_dim (~100) + Map_dim (225)
    expected_min = 300
    if obs.shape[0] > expected_min:
        print(f"✅ Obs dimension ({obs.shape[0]}) is > {expected_min}. Map seems included.")
    else:
        print(f"❌ Obs dimension ({obs.shape[0]}) is too small. Map might be missing.")
        return

    # 2. Create Network
    print("\n[2] Creating ActorCritic Network...")
    model = ActorCritic(
        obs_dim=obs.shape[0],
        act_dim=env.action_space.n,
        hidden_sizes=(128, 128),
        feat_dim=128
    )
    
    # Check if HybridBackbone is used
    if "HybridBackbone" in str(type(model.backbone)):
        print("✅ HybridBackbone is correctly initialized.")
    else:
        print(f"❌ Backbone type is {type(model.backbone)}. Expected HybridBackbone.")
        # return # Let's try forward anyway

    # 3. Forward Pass
    print("\n[3] Testing Forward Pass...")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) # (1, obs_dim)
    
    try:
        logits, value = model(obs_tensor)
        print(f"✅ Forward pass successful.")
        print(f"Logits shape: {logits.shape} (Expected: [1, 5])")
        print(f"Value shape: {value.shape} (Expected: [1, 1])")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cnn_integration()
