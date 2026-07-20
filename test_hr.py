import traceback
from agents.hr_agent import ask_hr
try:
    print("Testing ask_hr...")
    res = ask_hr('How many employees are in the Engineering department?')
    print("Success! First 100 chars:", res[:100])
except Exception as e:
    print("Exception!")
    print(traceback.format_exc())
