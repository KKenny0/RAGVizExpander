from typing import List


class TxtLoader:
    def load_data(self, file: str) -> List[str]:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        return [text]
