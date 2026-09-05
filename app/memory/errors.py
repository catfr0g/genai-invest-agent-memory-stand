class MemoryResetError(RuntimeError):
    """Ошибка reset с информацией о уже удалённых записях."""

    def __init__(self, deleted: dict[str, int], failed_at: str):
        self.deleted = deleted
        self.failed_at = failed_at
        super().__init__(f"Memory reset failed at {failed_at}")
