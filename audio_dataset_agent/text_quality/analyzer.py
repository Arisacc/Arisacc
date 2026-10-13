"""使用确定性编辑距离评价参考文本与 ASR 候选文本。"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class TextAlignmentResult:
    normalized_reference: str
    normalized_candidate: str
    character_errors: int
    cer: float
    word_errors: int
    wer: float
    similarity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_reference": self.normalized_reference,
            "normalized_candidate": self.normalized_candidate,
            "character_errors": self.character_errors,
            "cer": self.cer,
            "word_errors": self.word_errors,
            "wer": self.wer,
            "similarity": self.similarity,
        }


def normalize_text(text: str) -> str:
    """统一全半角和大小写，只保留字母与数字供 CER 比较。"""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character).startswith(("L", "N"))
    )


def evaluate_text_alignment(
    reference_text: str,
    candidate_text: str,
) -> TextAlignmentResult:
    normalized_reference = normalize_text(reference_text)
    normalized_candidate = normalize_text(candidate_text)
    character_errors = levenshtein_distance(
        list(normalized_reference),
        list(normalized_candidate),
    )
    reference_words = _tokenize_words(reference_text)
    candidate_words = _tokenize_words(candidate_text)
    word_errors = levenshtein_distance(reference_words, candidate_words)
    cer = character_errors / max(len(normalized_reference), 1)
    wer = word_errors / max(len(reference_words), 1)
    return TextAlignmentResult(
        normalized_reference=normalized_reference,
        normalized_candidate=normalized_candidate,
        character_errors=character_errors,
        cer=cer,
        word_errors=word_errors,
        wer=wer,
        similarity=max(0.0, 1.0 - cer),
    )


def levenshtein_distance(reference: Sequence[T], candidate: Sequence[T]) -> int:
    """用滚动数组计算编辑距离，内存复杂度为 O(min(m, n))。"""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for candidate_index, candidate_item in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1] + (ref_item != candidate_item),
                )
            )
        previous = current
    return previous[-1]


def _tokenize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # 中文按字、拉丁字母和数字按连续词元处理，避免依赖外部分词模型。
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", normalized)
