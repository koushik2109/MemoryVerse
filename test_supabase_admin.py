import asyncio
from supabase import create_client, Client
import os

from dotenv import load_dotenv
load_dotenv('backend/.env')

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)

try:
    print("Creating user...")
    res = supabase.auth.admin.create_user({
        "email": "test_admin_user@example.com",
        "password": "Password123!",
        "email_confirm": False
    })
    uid = res.user.id
    print(f"User created: {uid}, email_confirmed_at: {res.user.email_confirmed_at}")
    
    print("\nUpdating email_confirm=True...")
    res2 = supabase.auth.admin.update_user_by_id(uid, {"email_confirm": True})
    print(f"Updated email_confirmed_at: {res2.user.email_confirmed_at}")
    
    print("\nGenerating signup link...")
    link_res = supabase.auth.admin.generate_link({
        "type": "signup",
        "email": "test_admin_user@example.com",
        "password": "Password123!"
    })
    print(f"Generated link: {link_res.properties.action_link}")

    print("\nCleaning up...")
    supabase.auth.admin.delete_user(uid)
    print("Deleted.")

except Exception as e:
    print(f"Error: {e}")

