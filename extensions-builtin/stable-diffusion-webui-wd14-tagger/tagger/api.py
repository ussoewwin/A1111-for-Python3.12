"""API module for FastAPI"""
from typing import Callable, Dict, Optional
from threading import Lock
from secrets import compare_digest
import asyncio
from collections import defaultdict
from hashlib import sha256
import string
from random import choices

from modules import shared  # pylint: disable=import-error
from modules.api.api import decode_base64_to_image  # pylint: disable=E0401
from modules.call_queue import queue_lock  # pylint: disable=import-error
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from tagger import utils  # pylint: disable=import-error
from tagger import api_models as models  # pylint: disable=import-error


class Api:
    """Api class for FastAPI"""
    def __init__(
        self, app: FastAPI, qlock: Lock, prefix: Optional[str] = None
    ) -> None:
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.app = app
        self.queue: Dict[str, asyncio.Queue] = {}
        self.res: Dict[str, Dict[str, Dict[str, float]]] = \
            defaultdict(dict)
        self.queue_lock = qlock
        self.tasks: Dict[str, asyncio.Task] = {}

        self.runner: Optional[asyncio.Task] = None
        self.prefix = prefix
        self.running_batches: Dict[str, Dict[str, float]] = \
            defaultdict(lambda: defaultdict(int))

        self.add_api_route(
            'interrogate',
            self.endpoint_interrogate,
            methods=['POST'],
            response_model=models.TaggerInterrogateResponse
        )

        self.add_api_route(
            'interrogators',
            self.endpoint_interrogators,
            methods=['GET'],
            response_model=models.TaggerInterrogatorsResponse
        )

        self.add_api_route(
            'unload-interrogators',
            self.endpoint_unload_interrogators,
            methods=['POST'],
            response_model=str,
        )

    async def add_to_queue(self, m, q, n='', i=None, t=0.0) -> Dict[
        str, Dict[str, float]
    ]:
        if m not in self.queue:
            self.queue[m] = asyncio.Queue()
        #  loop = asyncio.get_running_loop()
        #  asyncio.run_coroutine_threadsafe(
        task = asyncio.create_task(self.queue[m].put((q, n, i, t)))
        #  , loop)

        if self.runner is None:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(self.batch_process(), loop=loop)
        await task
        return await self.tasks[q+"\t"+n]

    async def do_queued_interrogation(self, m, q, n, i, t) -> Dict[
        str, Dict[str, float]
    ]:
        self.running_batches[m][q] += 1.0
        # queue and name empty to process, not queue
        res = self.endpoint_interrogate(
            models.TaggerInterrogateRequest(
                image=i,
                model=m,
                threshold=t,
                name_in_queue='',
                queue=''
            )
        )
        self.res[q][n] = res.caption["tag"]
        for k, v in res.caption["rating"].items():
            self.res[q][n]["rating:"+k] = v
        return self.running_batches

    async def finish_queue(self, m, q) -> Dict[str, Dict[str, float]]:
        if q in self.running_batches[m]:
            del self.running_batches[m][q]
        if q in self.res:
            return self.res.pop(q)
        return self.running_batches

    async def batch_process(self) -> None:
        #  loop = asyncio.get_running_loop()
        while len(self.queue) > 0:
            for m in self.queue:
                # if zero the queue might just be pending
                while True:
                    try:
                        #  q, n, i, t = asyncio.run_coroutine_threadsafe(
                        #  self.queue[m].get_nowait(), loop).result()
                        q, n, i, t = self.queue[m].get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    self.tasks[q+"\t"+n] = asyncio.create_task(
                        self.do_queued_interrogation(m, q, n, i, t) if n != ""
                        else self.finish_queue(m, q)
                    )

            for model in self.running_batches:
                if len(self.running_batches[model]) == 0:
                    del self.queue[model]
            else:
                await asyncio.sleep(0.1)

        self.running_batches.clear()
        self.runner = None

    def auth(self, creds: Optional[HTTPBasicCredentials] = None):
        if creds is None:
            creds = Depends(HTTPBasic())
        if creds.username in self.credentials:
            if compare_digest(creds.password,
                              self.credentials[creds.username]):
                return True

        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Basic"
            })

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        if self.prefix:
            path = f'{self.prefix}/{path}'

        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[
                   Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    async def queue_interrogation(self, m, q, n='', i=None, t=0.0) -> Dict[
        str, Dict[str, float]
    ]:
        """ queue an interrogation, or add to batch """
        if n == '':
            task = asyncio.create_task(self.add_to_queue(m, q))
        else:
            if n == '<sha256>':
                n = sha256(i).hexdigest()
                if n in self.res[q]:
                    return self.running_batches
            elif n in self.res[q]:
                # clobber name if it's already in the queue
                j = 0
                while f'{n}#{j}' in self.res[q]:
                    j += 1
                n = f'{n}#{j}'
            self.res[q][n] = {}
            # add image to queue
            task = asyncio.create_task(self.add_to_queue(m, q, n, i, t))
        return await task

    def endpoint_interrogate(self, req: models.TaggerInterrogateRequest):
        """ one file interrogation, queueing, or batch results """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators:
            raise HTTPException(404, 'Model not found')

        m, q, n = (req.model, req.queue, req.name_in_queue)
        res: Dict[str, Dict[str, float]] = {}

        if q != '' or n != '':
            if q == '':
                # generate a random queue name, not in use
                while True:
                    q = ''.join(choices(string.ascii_uppercase +
                                string.digits, k=8))
                    if q not in self.queue:
                        break
                print(f'WD14 tagger api generated queue name: {q}')
            res = asyncio.run(self.queue_interrogation(m, q, n, req.image,
                              req.threshold), debug=True)
        else:
            image = decode_base64_to_image(req.image)
            interrogator = utils.interrogators[m]
            res = {"tag": {}, "rating": {}}
            with self.queue_lock:
                res["rating"], tag = interrogator.interrogate(image)

            for k, v in tag.items():
                if v > req.threshold:
                    res["tag"][k] = v

        return models.TaggerInterrogateResponse(caption=res)

    def endpoint_interrogators(self):
        return models.TaggerInterrogatorsResponse(
            models=list(utils.interrogators.keys())
        )

    def endpoint_unload_interrogators(self):
        unloaded_models = 0

        for i in utils.interrogators.values():
            if i.unload():
                unloaded_models = unloaded_models + 1

        return f"Successfully unload {unloaded_models} model(s)"


def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/tagger/v1')
