from process_worker import ProcessWorker
from pathlib import Path
from queue import Queue
import websockets
import asyncio
import json
import logging


logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

image_queue = Queue()
done_dict = dict()


async def process_done(image_id):
    while True:
        if image_id in done_dict:
            return True
        await asyncio.sleep(1)


async def echo(websocket, path):
    async for message in websocket:
        msg = json.loads(message)
        print('Got message:', msg)
        if 'event' in msg:
            if (msg['event'] == 'image_process') and ('image_id' in msg):
                image_id = msg['image_id']
                image_queue.put_nowait(image_id)
                await websocket.send(json.dumps({'event': 'put_to_queue'}))
                await process_done(image_id)
                await websocket.send(json.dumps({'event': 'processing_done', 'image_id': image_id, 'result': done_dict[image_id]}))
                del done_dict[image_id]
            elif msg['event'] == 'ping':
                await websocket.send(json.dumps({'event': 'pong'}))

if __name__ == '__main__':
    worker = ProcessWorker(image_queue, done_dict, '../michigan_frontend2/public/results')
    worker.start()
    print('Worker started.')
    asyncio.get_event_loop().run_until_complete(websockets.serve(echo, '0.0.0.0', 8766))
    asyncio.get_event_loop().run_forever()
