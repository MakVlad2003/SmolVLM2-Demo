import threading
import queue
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class ResultBroker:
    def __init__(self) -> None:
        self.incoming: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._waiters: Dict[int, "queue.Queue[Dict[str, Any]]"] = {}
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[ResultBroker] Started ✅")

    def register(self, task_id: int) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        with self._lock:
            if task_id in self._pending:
                q.put(self._pending.pop(task_id))
            else:
                self._waiters[task_id] = q
        return q

    def _loop(self) -> None:
        while True:
            result = self.incoming.get()
            try:
                task_id = result.get("id")
                if task_id is None:
                    logger.warning("[ResultBroker] Got result without 'id' key, ignoring")
                    continue

                with self._lock:
                    waiter = self._waiters.pop(task_id, None)
                    if waiter is not None:
                        waiter.put(result)
                    else:
                        self._pending[task_id] = result
            finally:
                self.incoming.task_done()
