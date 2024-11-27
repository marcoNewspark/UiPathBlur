import json
def test():
    result_dict = {
        "total_images": 0,
        "total_faces": 1,
        "files": [{"input": "C:\data", "output": "C:\data"}]
    }

    return json.dumps(result_dict)