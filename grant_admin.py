import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from services.supabase_client import get_supabase_admin_client

def grant_admin():
    user_id = "4371c552-2e1c-494b-a152-10929beeab81"
    print(f"Granting admin role to UID: {user_id}...")
    
    try:
        client = get_supabase_admin_client()
        
        # Upsert profile
        res = client.table("profiles").upsert({
            "id": user_id,
            "role": "admin",
            "full_name": "Admin User"
        }).execute()
            
        print("✅ SUCCESS: Admin role granted.")

    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    grant_admin()
