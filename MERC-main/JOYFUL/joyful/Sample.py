from typing import Optional


_SBERT_MODEL: Optional[object] = None


def _get_sbert_model():
    """
    惰性加载 SentenceTransformer：避免在 unpickle 时强依赖 torch / 下载模型。
    训练/构造新 Sample 时才会真正初始化模型。
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SBERT_MODEL = SentenceTransformer("paraphrase-distilroberta-base-v1")
    return _SBERT_MODEL


class Sample:
    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence
        self.sbert_sentence_embeddings = _get_sbert_model().encode(sentence)
