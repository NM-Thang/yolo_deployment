import requests
import cv2
import numpy as np
import asyncio
import websockets
import json
import base64



# url = "http://123.30.171.173:8080/video"
# stream = requests.get(url, stream=True)

# bytes_data = b""
# for chunk in stream.iter_content(chunk_size=1024):
#     bytes_data += chunk
#     a = bytes_data.find(b'\xff\xd8')  # JPEG start
#     b = bytes_data.find(b'\xff\xd9')  # JPEG end
#     if a != -1 and b != -1:
#         jpg = bytes_data[a:b+2]
#         bytes_data = bytes_data[b+2:]
#         img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#         if img is not None:
#             cv2.imshow("Stream", img)
#             if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):  # Adjust delay based on desired frame rate
#                 break
# cv2.destroyAllWindows()


async def receive_video():
    uri = "ws://localhost:8080/ws/video"
    async with websockets.connect(uri) as websocket:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            frame_data = base64.b64decode(data["frame"])
            np_arr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            event = data["event"]
            for e in event:
                print(e)
                
            cv2.imshow("Video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

asyncio.run(receive_video())