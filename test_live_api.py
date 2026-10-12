import urllib.request
import json

url = "https://policypilot-uvj4.onrender.com/ask"
payload = {"question": "How many days of annual leave do employees receive?"}
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Origin": "https://policy-pilot-eta.vercel.app",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=90) as response:
        print("HTTP:", response.status)
        print(response.read().decode("utf-8")[:1500])
except Exception as error:
    print("Error:", repr(error))
    if hasattr(error, "read"):
        print(error.read().decode("utf-8")[:1500])
