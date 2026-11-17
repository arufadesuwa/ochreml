"""
OchreML Text Feature Extraction Submodule (*≧ω≦*)
Implements CountVectorizer, TfidfTransformer, TfidfVectorizer, and HashingVectorizer
backed by Rust Rayon tokenization, vocabulary indexing, TF-IDF scaling, and MurmurHash3.
"""

import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .._ochreml import (
    CountVectorizer as _RustCountVectorizer,
    HashingVectorizer as _RustHashingVectorizer,
    TfidfTransformer as _RustTfidfTransformer,
)
from .._preproc_utils import (
    bytes_to_f64_2d,
    csr_bytes_to_scipy,
    load_dit_envelope,
    save_dit_envelope,
    validate_dit_path,
)

_TAG_TFIDF_VECTORIZER = 32

ENGLISH_STOP_WORDS = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always", "am", "among",
    "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone",
    "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
    "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could",
    "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during",
    "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
    "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except",
    "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former",
    "formerly", "forty", "found", "four", "from", "front", "full", "further", "get",
    "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here",
    "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself",
    "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile",
    "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
    "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless",
    "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now",
    "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or",
    "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
    "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem",
    "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten",
    "than", "that", "the", "their", "them", "themselves", "then", "thence", "there",
    "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
    "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
    "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was",
    "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where",
    "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
    "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why",
    "will", "with", "within", "without", "would", "yet", "you", "your", "yours",
    "yourself", "yourselves",
})


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    from .. import get_default_device

    return get_default_device()


def _resolve_stop_words(stop_words: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if stop_words is None:
        return []
    if stop_words == "english":
        return sorted(ENGLISH_STOP_WORDS)
    return [str(w) for w in stop_words]


def _strip_accents_unicode(text: str) -> str:
    nkfd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nkfd if not unicodedata.combining(c))


def _strip_accents_ascii(text: str) -> str:
    nkfd = unicodedata.normalize("NFKD", text)
    return nkfd.encode("ascii", "ignore").decode("ascii")


def _read_docs(
    raw_documents: Iterable[Any], input_mode: str, encoding: str, decode_error: str
) -> List[str]:
    docs: List[str] = []
    for item in raw_documents:
        if input_mode == "filename":
            with open(item, "rb") as f:
                raw = f.read()
            docs.append(raw.decode(encoding, decode_error))
        elif input_mode == "file":
            raw = item.read()
            docs.append(raw.decode(encoding, decode_error) if isinstance(raw, bytes) else str(raw))
        else:
            docs.append(item.decode(encoding, decode_error) if isinstance(item, bytes) else str(item))
    return docs


class _TextVectorizerMixin:
    """Shared text preprocessing, custom analyzer fallback, and stop-words helpers (＾▽＾)."""

    input: str
    encoding: str
    decode_error: str
    strip_accents: Optional[Union[str, Callable[..., str]]]
    lowercase: bool
    preprocessor: Optional[Callable[[str], str]]
    tokenizer: Optional[Callable[[str], List[str]]]
    analyzer: Union[str, Callable[[str], List[str]]]
    token_pattern: Optional[str]
    ngram_range: Tuple[int, int]
    stop_words: Optional[Union[str, Iterable[str]]]

    def _can_use_rust_analyzer(self) -> bool:
        return (
            self.strip_accents is None
            and self.preprocessor is None
            and self.tokenizer is None
            and isinstance(self.analyzer, str)
            and (
                self.analyzer in ("char", "char_wb")
                or self.token_pattern in (r"(?u)\b\w\w+\b", r"\b\w\w+\b", None)
            )
        )

    def build_preprocessor(self) -> Callable[[str], str]:
        """Return a function to preprocess the text before tokenization (＾▽＾)."""
        if self.preprocessor is not None:
            return self.preprocessor

        def _pre(doc: str) -> str:
            if self.lowercase:
                doc = doc.lower()
            if self.strip_accents == "unicode":
                doc = _strip_accents_unicode(doc)
            elif self.strip_accents == "ascii":
                doc = _strip_accents_ascii(doc)
            elif callable(self.strip_accents):
                doc = self.strip_accents(doc)
            return doc

        return _pre

    def build_tokenizer(self) -> Callable[[str], List[str]]:
        """Return a function that splits a string into a sequence of tokens (*≧ω≦*)."""
        if self.tokenizer is not None:
            return self.tokenizer
        pat = re.compile(self.token_pattern or r"(?u)\b\w\w+\b")
        return pat.findall

    def build_analyzer(self) -> Callable[[str], List[str]]:
        """Return a callable to process input data into a sequence of n-grams (o´∀｀o)."""
        if callable(self.analyzer):
            return self.analyzer
        pre = self.build_preprocessor()
        stop_set = set(_resolve_stop_words(self.stop_words))
        min_n, max_n = self.ngram_range

        if self.analyzer == "word":
            tok = self.build_tokenizer()

            def _analyze_word(doc: str) -> List[str]:
                tokens = [t for t in tok(pre(doc)) if t not in stop_set]
                if min_n == 1 and max_n == 1:
                    return tokens
                out: List[str] = []
                n_tok = len(tokens)
                for n in range(min_n, min(max_n, n_tok) + 1):
                    if n == 1:
                        out.extend(tokens)
                    else:
                        out.extend(" ".join(tokens[i : i + n]) for i in range(n_tok - n + 1))
                return out

            return _analyze_word

        if self.analyzer == "char":

            def _analyze_char(doc: str) -> List[str]:
                s = " ".join(pre(doc).split())
                out: List[str] = []
                for n in range(min_n, max_n + 1):
                    out.extend(s[i : i + n] for i in range(max(0, len(s) - n + 1)))
                return out

            return _analyze_char

        def _analyze_char_wb(doc: str) -> List[str]:
            s = " ".join(pre(doc).split())
            out: List[str] = []
            for w in s.split():
                padded = f" {w} "
                for n in range(min_n, max_n + 1):
                    out.extend(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))
            return out

        return _analyze_char_wb


class CountVectorizer(_TextVectorizerMixin):
    """
    Convert a collection of text documents to a matrix of token counts (o´∀｀o)
    """

    _rust_cls = _RustCountVectorizer

    def __init__(
        self,
        *,
        input: str = "content",
        encoding: str = "utf-8",
        decode_error: str = "strict",
        strip_accents: Optional[Union[str, Callable[..., str]]] = None,
        lowercase: bool = True,
        preprocessor: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        stop_words: Optional[Union[str, Iterable[str]]] = None,
        token_pattern: Optional[str] = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        analyzer: Union[str, Callable[[str], List[str]]] = "word",
        max_df: Union[int, float] = 1.0,
        min_df: Union[int, float] = 1,
        max_features: Optional[int] = None,
        vocabulary: Optional[Union[Dict[str, int], Iterable[str]]] = None,
        binary: bool = False,
        dtype: Any = np.int64,
        device: Optional[str] = None,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
        self.device = _resolve_device(device)

        fixed_vocab = None
        if vocabulary is not None:
            if isinstance(vocabulary, dict):
                fixed_vocab = {str(k): int(v) for k, v in vocabulary.items()}
            else:
                fixed_vocab = {str(k): i for i, k in enumerate(sorted(set(vocabulary)))}

        self._model = _RustCountVectorizer(
            lowercase=lowercase,
            analyzer=analyzer if isinstance(analyzer, str) else "word",
            ngram_range=self.ngram_range,
            max_df=float(max_df),
            min_df=float(min_df),
            max_features=max_features,
            binary=binary,
            stop_words=_resolve_stop_words(stop_words),
            fixed_vocab=fixed_vocab,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustCountVectorizer, device: str) -> "CountVectorizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.input = "content"
        inst.encoding = "utf-8"
        inst.decode_error = "strict"
        inst.strip_accents = None
        inst.lowercase = True
        inst.preprocessor = None
        inst.tokenizer = None
        inst.stop_words = None
        inst.token_pattern = r"(?u)\b\w\w+\b"
        inst.ngram_range = (1, 1)
        inst.analyzer = "word"
        inst.max_df = 1.0
        inst.min_df = 1
        inst.max_features = None
        inst.vocabulary = None
        inst.binary = False
        inst.dtype = np.int64
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustCountVectorizer.from_bytes(state, target_dev)
        self.input = "content"
        self.encoding = "utf-8"
        self.decode_error = "strict"
        self.strip_accents = None
        self.lowercase = True
        self.preprocessor = None
        self.tokenizer = None
        self.stop_words = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.ngram_range = (1, 1)
        self.analyzer = "word"
        self.max_df = 1.0
        self.min_df = 1
        self.max_features = None
        self.vocabulary = None
        self.binary = False
        self.dtype = np.int64
        self.device = self._model.device_ or target_dev

    def fit(self, raw_documents: Iterable[Any], y: Optional[Any] = None) -> "CountVectorizer":
        """Learn a vocabulary dictionary of all tokens in the raw documents (*≧ω≦*)."""
        docs = _read_docs(raw_documents, self.input, self.encoding, self.decode_error)
        if self._can_use_rust_analyzer():
            self._model.fit_docs(docs)
        else:
            analyze = self.build_analyzer()
            self._model.fit_tokenized([analyze(d) for d in docs])
        return self

    def transform(self, raw_documents: Iterable[Any]) -> Any:
        """Transform documents to document-term matrix (＾▽＾)."""
        docs = _read_docs(raw_documents, self.input, self.encoding, self.decode_error)
        if self._can_use_rust_analyzer():
            parts = self._model.transform_docs_csr(docs)
        else:
            analyze = self.build_analyzer()
            parts = self._model.transform_tokenized_csr([analyze(d) for d in docs])
        return csr_bytes_to_scipy(parts, dtype=self.dtype)

    def fit_transform(self, raw_documents: Iterable[Any], y: Optional[Any] = None) -> Any:
        """Learn the vocabulary dictionary and return document-term matrix (o´∀｀o)."""
        docs = _read_docs(raw_documents, self.input, self.encoding, self.decode_error)
        return self.fit(docs, y).transform(docs)

    def inverse_transform(self, X: Any) -> List[np.ndarray]:
        """Return terms per document with nonzero entries in X (＾▽＾)."""
        vocab = self.vocabulary_
        if vocab is None:
            raise ValueError("CountVectorizer is not fitted yet (´-ω-`)")
        inv_vocab = [""] * len(vocab)
        for term, idx in vocab.items():
            inv_vocab[idx] = term
        inv_arr = np.asarray(inv_vocab, dtype=object)
        if hasattr(X, "tocsr"):
            csr = X.tocsr()
            return [inv_arr[csr.indices[csr.indptr[r] : csr.indptr[r + 1]]] for r in range(csr.shape[0])]
        arr = np.asarray(X)
        return [inv_arr[np.nonzero(row)[0]] for row in arr]

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Get output feature names for transformation (＾▽＾)."""
        vocab = self.vocabulary_
        if vocab is None:
            raise ValueError("CountVectorizer is not fitted yet (´-ω-`)")
        names = [""] * len(vocab)
        for term, idx in vocab.items():
            names[idx] = term
        return np.asarray(names, dtype=object)

    @property
    def vocabulary_(self) -> Optional[Dict[str, int]]:
        return self._model.vocabulary_

    @property
    def fixed_vocabulary_(self) -> bool:
        return self._model.fixed_vocabulary_

    @property
    def stop_words_(self) -> set:
        return set(self._model.stop_words_)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CountVectorizer":
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustCountVectorizer.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "CountVectorizer":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"CountVectorizer(ngram_range={self.ngram_range}, analyzer={self.analyzer!r}) (o´∀｀o)"


class TfidfTransformer:
    """
    Transform a count matrix to a normalized tf or tf-idf representation (＾▽＾)
    """

    _rust_cls = _RustTfidfTransformer

    def __init__(
        self,
        *,
        norm: Optional[str] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        device: Optional[str] = None,
    ):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.device = _resolve_device(device)
        self._model = _RustTfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustTfidfTransformer, device: str) -> "TfidfTransformer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.norm = "l2"
        inst.use_idf = True
        inst.smooth_idf = True
        inst.sublinear_tf = False
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustTfidfTransformer.from_bytes(state, target_dev)
        self.norm = "l2"
        self.use_idf = True
        self.smooth_idf = True
        self.sublinear_tf = False
        self.device = self._model.device_ or target_dev

    def fit(self, X: Any, y: Optional[Any] = None) -> "TfidfTransformer":
        """Learn the idf vector (global term weights) (*≧ω≦*)."""
        if hasattr(X, "tocsr"):
            csr = X.tocsr()
            self._model.fit_csr(
                csr.data.astype(np.float64).tolist(),
                csr.indices.astype(np.int64).tolist(),
                csr.indptr.astype(np.int64).tolist(),
                csr.shape[0],
                csr.shape[1],
            )
        else:
            arr = np.ascontiguousarray(X, dtype=np.float64)
            self._model.fit_buffer(arr)
        return self

    def transform(self, X: Any, copy: bool = True) -> Any:
        """Transform a count matrix to a tf or tf-idf representation (o´∀｀o)."""
        if hasattr(X, "tocsr"):
            csr = X.tocsr()
            parts = self._model.transform_csr(
                csr.data.astype(np.float64).tolist(),
                csr.indices.astype(np.int64).tolist(),
                csr.indptr.astype(np.int64).tolist(),
                csr.shape[0],
                csr.shape[1],
            )
            return csr_bytes_to_scipy(parts, dtype=np.float64)
        arr = np.ascontiguousarray(X, dtype=np.float64)
        rows, cols, raw = self._model.transform_buffer(arr)
        return bytes_to_f64_2d(rows, cols, raw)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """Fit and transform X (＾▽＾)."""
        return self.fit(X, y).transform(X)

    @property
    def idf_(self) -> Optional[np.ndarray]:
        v = self._model.idf_
        return None if v is None else np.asarray(v, dtype=np.float64)

    @property
    def n_features_in_(self) -> Optional[int]:
        return self._model.n_features_in_

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TfidfTransformer":
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustTfidfTransformer.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "TfidfTransformer":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"TfidfTransformer(norm={self.norm!r}, use_idf={self.use_idf}) (＾▽＾)"


class TfidfVectorizer(CountVectorizer):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features (*≧ω≦*)
    """

    def __init__(
        self,
        *,
        input: str = "content",
        encoding: str = "utf-8",
        decode_error: str = "strict",
        strip_accents: Optional[Union[str, Callable[..., str]]] = None,
        lowercase: bool = True,
        preprocessor: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        analyzer: Union[str, Callable[[str], List[str]]] = "word",
        stop_words: Optional[Union[str, Iterable[str]]] = None,
        token_pattern: Optional[str] = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        max_df: Union[int, float] = 1.0,
        min_df: Union[int, float] = 1,
        max_features: Optional[int] = None,
        vocabulary: Optional[Union[Dict[str, int], Iterable[str]]] = None,
        binary: bool = False,
        dtype: Any = np.float64,
        norm: Optional[str] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            device=device,
        )
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self._tfidf = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            device=self.device,
        )

    def fit(self, raw_documents: Iterable[Any], y: Optional[Any] = None) -> "TfidfVectorizer":
        """Learn vocabulary and idf from training set (＾▽＾)."""
        docs = _read_docs(raw_documents, self.input, self.encoding, self.decode_error)
        CountVectorizer.fit(self, docs)
        X_cnt = CountVectorizer.transform(self, docs)
        self._tfidf.fit(X_cnt)
        return self

    def transform(self, raw_documents: Iterable[Any]) -> Any:
        """Transform documents to document-term TF-IDF matrix (*≧ω≦*)."""
        X_cnt = CountVectorizer.transform(self, raw_documents)
        res = self._tfidf.transform(X_cnt)
        if np.dtype(self.dtype) != np.float64:
            res = res.astype(self.dtype)
        return res

    def fit_transform(self, raw_documents: Iterable[Any], y: Optional[Any] = None) -> Any:
        """Learn vocabulary and idf, return document-term matrix (o´∀｀o)."""
        docs = _read_docs(raw_documents, self.input, self.encoding, self.decode_error)
        CountVectorizer.fit(self, docs)
        X_cnt = CountVectorizer.transform(self, docs)
        res = self._tfidf.fit_transform(X_cnt)
        if np.dtype(self.dtype) != np.float64:
            res = res.astype(self.dtype)
        return res

    @property
    def idf_(self) -> Optional[np.ndarray]:
        return self._tfidf.idf_

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "cv_bytes": self._model.to_bytes(),
            "tfidf_bytes": self._tfidf._model.to_bytes(),
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "dtype": self.dtype,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustCountVectorizer.from_bytes(state["cv_bytes"], target_dev)
        self._tfidf = TfidfTransformer._from_rust(
            _RustTfidfTransformer.from_bytes(state["tfidf_bytes"], target_dev), target_dev
        )
        self.input = "content"
        self.encoding = "utf-8"
        self.decode_error = "strict"
        self.strip_accents = None
        self.lowercase = True
        self.preprocessor = None
        self.tokenizer = None
        self.stop_words = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.ngram_range = (1, 1)
        self.analyzer = "word"
        self.max_df = 1.0
        self.min_df = 1
        self.max_features = None
        self.vocabulary = None
        self.binary = False
        self.dtype = state.get("dtype", np.float64)
        self.norm = state.get("norm", "l2")
        self.use_idf = state.get("use_idf", True)
        self.smooth_idf = state.get("smooth_idf", True)
        self.sublinear_tf = state.get("sublinear_tf", False)
        self.device = self._model.device_ or target_dev

    def save(self, path: str) -> None:
        save_dit_envelope(_TAG_TFIDF_VECTORIZER, self.__getstate__(), path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TfidfVectorizer":
        with open(validate_dit_path(path), "rb") as f:
            state = load_dit_envelope(_TAG_TFIDF_VECTORIZER, f.read())
        inst = cls.__new__(cls)
        inst.__setstate__(state)
        if device is not None:
            inst.to(device)
        return inst

    def __repr__(self) -> str:
        return f"TfidfVectorizer(ngram_range={self.ngram_range}, norm={self.norm!r}) (*≧ω≦*)"


class HashingVectorizer(_TextVectorizerMixin):
    """
    Convert a collection of text documents to a matrix of token occurrences via MurmurHash3 (*≧ω≦*)
    """

    _rust_cls = _RustHashingVectorizer

    def __init__(
        self,
        *,
        input: str = "content",
        encoding: str = "utf-8",
        decode_error: str = "strict",
        strip_accents: Optional[Union[str, Callable[..., str]]] = None,
        lowercase: bool = True,
        preprocessor: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        stop_words: Optional[Union[str, Iterable[str]]] = None,
        token_pattern: Optional[str] = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        analyzer: Union[str, Callable[[str], List[str]]] = "word",
        n_features: int = 2**20,
        binary: bool = False,
        norm: Optional[str] = "l2",
        alternate_sign: bool = True,
        dtype: Any = np.float64,
        device: Optional[str] = None,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.analyzer = analyzer
        self.n_features = n_features
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.dtype = dtype
        self.device = _resolve_device(device)
        self._model = _RustHashingVectorizer(
            lowercase=lowercase,
            analyzer=analyzer if isinstance(analyzer, str) else "word",
            ngram_range=self.ngram_range,
            n_features=n_features,
            binary=binary,
            norm=norm,
            alternate_sign=alternate_sign,
            stop_words=_resolve_stop_words(stop_words),
            device=self.device,
        )

    @classmethod
    def _from_rust(cls, rust_obj: _RustHashingVectorizer, device: str) -> "HashingVectorizer":
        inst = cls.__new__(cls)
        inst._model = rust_obj
        inst.input = "content"
        inst.encoding = "utf-8"
        inst.decode_error = "strict"
        inst.strip_accents = None
        inst.lowercase = True
        inst.preprocessor = None
        inst.tokenizer = None
        inst.stop_words = None
        inst.token_pattern = r"(?u)\b\w\w+\b"
        inst.ngram_range = (1, 1)
        inst.analyzer = "word"
        inst.n_features = rust_obj.n_features
        inst.binary = False
        inst.norm = "l2"
        inst.alternate_sign = True
        inst.dtype = np.float64
        inst.device = rust_obj.device_ or device
        return inst

    def __getstate__(self) -> bytes:
        return self._model.to_bytes()

    def __setstate__(self, state: bytes) -> None:
        target_dev = _resolve_device(None)
        self._model = _RustHashingVectorizer.from_bytes(state, target_dev)
        self.input = "content"
        self.encoding = "utf-8"
        self.decode_error = "strict"
        self.strip_accents = None
        self.lowercase = True
        self.preprocessor = None
        self.tokenizer = None
        self.stop_words = None
        self.token_pattern = r"(?u)\b\w\w+\b"
        self.ngram_range = (1, 1)
        self.analyzer = "word"
        self.n_features = self._model.n_features
        self.binary = False
        self.norm = "l2"
        self.alternate_sign = True
        self.dtype = np.float64
        self.device = self._model.device_ or target_dev

    def fit(self, X: Optional[Any] = None, y: Optional[Any] = None) -> "HashingVectorizer":
        """Only validates estimator's parameters (stateless) (＾▽＾)."""
        return self

    def partial_fit(self, X: Optional[Any] = None, y: Optional[Any] = None) -> "HashingVectorizer":
        """Only validates estimator's parameters (stateless) (o´∀｀o)."""
        return self

    def transform(self, X: Iterable[Any]) -> Any:
        """Transform a sequence of documents to a document-term matrix (*≧ω≦*)."""
        docs = _read_docs(X, self.input, self.encoding, self.decode_error)
        if self._can_use_rust_analyzer():
            parts = self._model.transform_docs_csr(docs)
        else:
            analyze = self.build_analyzer()
            parts = self._model.transform_tokenized_csr([analyze(d) for d in docs])
        return csr_bytes_to_scipy(parts, dtype=self.dtype)

    def fit_transform(self, X: Iterable[Any], y: Optional[Any] = None) -> Any:
        """Transform a sequence of documents to a document-term matrix (＾▽＾)."""
        return self.fit(X, y).transform(X)

    @property
    def device_(self) -> Optional[str]:
        return self._model.device_

    def save(self, path: str) -> None:
        self._model.save(validate_dit_path(path))

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "HashingVectorizer":
        target_dev = _resolve_device(device)
        return cls._from_rust(_RustHashingVectorizer.load(validate_dit_path(path), target_dev), target_dev)

    def to(self, device: str) -> "HashingVectorizer":
        self._model.set_device(device)
        self.device = self._model.device_ or device
        return self

    def __repr__(self) -> str:
        return f"HashingVectorizer(n_features={self.n_features}, norm={self.norm!r}) (*≧ω≦*)"


__all__ = [
    "CountVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
    "HashingVectorizer",
    "ENGLISH_STOP_WORDS",
]
