import threading
import queue
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class ResultBroker:
    """
    Небольшой роутер между воркером и многими потребителями (UI, API).
    - Воркер кладёт результаты в общую очередь `incoming`.
    - UI/API регистрируют интерес по `task_id` и получают отдельную очередь.
    """

    def __init__(self) -> None:
        self.incoming: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._waiters: Dict[int, "queue.Queue[Dict[str, Any]]"] = {}
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[ResultBroker] Started ✅")

    def register(self, task_id: int) -> "queue.Queue[Dict[str, Any]]":
        """
        Зарегистрировать ожидание результата для task_id.
        Возвращает очередь, в которую будет положен ровно один dict с результатом.
        Если результат уже пришёл раньше — вернём очередь с уже лежащим значением.
        """
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        with self._lock:
            if task_id in self._pending:
                # Результат уже пришёл до регистрации
                q.put(self._pending.pop(task_id))
            else:
                # Будем ждать
                self._waiters[task_id] = q
        return q

    def _loop(self) -> None:
        """Основной цикл: слушает общую очередь incoming и раздаёт результаты по task_id."""
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
                        # Никто пока не ждёт — запомним
                        self._pending[task_id] = result
            finally:
                self.incoming.task_done()
