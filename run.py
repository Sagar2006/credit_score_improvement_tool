from app import app
import sys

print("Starting Flask app")
print("sys.path:", sys.path)

if __name__ == '__main__':
    app.run(debug=True)