# handler.py

def handler(event):
    """
    Hàm chính nhận vào dữ liệu từ RunPod và trả về kết quả
    """
    try:
        input_data = event["input"]  # lấy input từ JSON
        text = input_data.get("text", "")

        # Bạn có thể chạy script bên ngoài, hoặc xử lý trực tiếp ở đây
        result = text.upper()  # Ví dụ: chuyển text thành chữ in hoa

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}
