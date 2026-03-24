## 0.0.6
+ Updated build script to export `DEP_FUNCTION_LIB_DIR` for path to the Function C library.

## 0.0.5
+ Added `muna.beta.openai.embeddings.create` method for using text embedding models via an OpenAI-compatible client.

## 0.0.4
+ Fixed 403 error when making predictions that have not been cached on the local disk.

## 0.0.3
+ Added `Serialize` derive to `RemotePrediction` and `RemotePredictionEvent`.

## 0.0.2
+ Updated `muna.predictions.create` method to allow concurrent multi-threaded usage.

## 0.0.1
+ First pre-release.