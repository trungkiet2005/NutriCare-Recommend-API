from pyngrok import ngrok, conf
import uvicorn
import socket
import time
import os
import subprocess
import signal
import requests
import threading



# Hàm kiểm tra xem cổng có đang được sử dụng không
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Hàm tìm cổng trống
def find_free_port(start_port=8000, max_attempts=10):
    port = start_port
    attempts = 0
    
    while attempts < max_attempts:
        if not is_port_in_use(port):
            return port
        port += 1
        attempts += 1
    
    raise RuntimeError(f"Không tìm thấy cổng trống sau {max_attempts} lần thử")

# Hàm giải phóng cổng nếu đang bị sử dụng (Linux/macOS)
def free_port(port):
    try:
        # Tìm process đang sử dụng cổng
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                              capture_output=True, text=True)
        
        # Nếu có process đang sử dụng
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Bỏ qua dòng header
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        print(f"Đang giải phóng process {pid} trên cổng {port}")
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)  # Chờ process kết thúc
                return True
    except Exception as e:
        print(f"Lỗi khi giải phóng cổng {port}: {e}")
    
    return False

# Ngắt tất cả tunnel đang mở
ngrok.kill()

# Tìm cổng trống
try:
    PORT = find_free_port()
    print(f"Tìm thấy cổng trống: {PORT}")
except:
    # Nếu không tìm được cổng trống, cố gắng giải phóng cổng 8000
    print("Không tìm thấy cổng trống, đang thử giải phóng cổng 8000...")
    if free_port(8000):
        PORT = 8000
    else:
        # Thử cổng 8080 nếu không giải phóng được 8000
        PORT = 8080
        if is_port_in_use(PORT):
            free_port(PORT)

print(f"Sử dụng cổng {PORT}")

# Mở tunnel ngrok
try:
    public_url = ngrok.connect(PORT)
    print(f"🌐 API public URL: {public_url}")
except Exception as e:
    print(f"Lỗi khi kết nối ngrok: {e}")
    exit(1)

# Chạy ứng dụng FastAPI trong thread riêng biệt
def run_app():
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
    except Exception as e:
        print(f"Lỗi khi chạy ứng dụng: {e}")

thread = threading.Thread(target=run_app)
thread.daemon = True  # Cho phép thoát chương trình khi thread còn chạy
thread.start()

# Đợi ứng dụng khởi động
print("Đợi ứng dụng khởi động...")
time.sleep(10)

# Kiểm tra ứng dụng đã chạy chưa
try:
    response = requests.get(f"http://localhost:{PORT}")
    print(f"Ứng dụng đã chạy, status code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Lỗi khi kiểm tra ứng dụng: {e}")
