import json
import urllib.request
import urllib.error


def main():
    url = "http://localhost:11434/api/tags"
    print(f"Connecting to Ollama at {url}...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PolicyPilot"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                print(f"Error: Ollama returned status code {response.status}")
                return

            data = json.loads(response.read().decode("utf-8"))
            models = data.get("models", [])

            print("Ollama connection successful.\n")
            print("Available models:")
            for m in models:
                name = m.get("name")
                size = m.get("size", 0) / (1024 * 1024 * 1024)
                print(f"- {name} ({size:.2f} GB)")

            if not models:
                print("Warning: No models currently installed in Ollama.")

    except urllib.error.URLError as e:
        print(f"Error: Failed to connect to Ollama at {url}. Make sure Ollama is running.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
