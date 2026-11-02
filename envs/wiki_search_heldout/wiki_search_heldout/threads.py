"""Cap the CPU threads of chroma's MiniLM title embedder in wiki tool servers.

chromadb builds its ONNX Runtime session with default options, so each tool server grabs every
physical core. A training run starts one tool server per env worker; on the shared 128-thread
server six of them held the load average above 300 while the GPUs waited on searches. A title
query is one short string, so two threads are plenty. Import this before chroma builds the
session (the tool server's setup).
"""

import os

import onnxruntime

THREADS = int(os.environ.get("WIKI_EMBED_THREADS", "2"))
_SessionOptions = onnxruntime.SessionOptions


def _limited_session_options() -> onnxruntime.SessionOptions:
    options = _SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    return options


# chromadb calls onnxruntime.SessionOptions() through the module, so replacing it here takes effect
onnxruntime.SessionOptions = _limited_session_options
